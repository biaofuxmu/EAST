import torch
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

from llamafactory.data.template import TEMPLATES, _add_or_replace_eos_token
from llamafactory.extras.logging import get_logger
import sacrebleu
from bleurt import score
from comet import load_from_checkpoint
from statistics import mean
import json
from latency import compute_delays, AverageLagging, LengthAdaptiveAverageLagging

logger = get_logger(__name__)


def tokenize_chinese(text, tokenizer):
    input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    idx = 0
    tok_ids = []
    tokens = []
    while idx < len(input_ids):
        tok_ids.append(input_ids[idx])
        token = tokenizer.decode(tok_ids)
        if "�" not in token:
            tokens.append(token)
            tok_ids = []
        idx += 1
    return tokens


class SimulInference:
    def __init__(self, args):
        self.args = args

        self.load_tokenizer_and_model(self.args.model_path)
        self.gen_kwargs = self.prepare_gen_kwargs(args)
        self.set_special_tokens()

        self.test_data = self.load_eval_datasets(self.args.data_path)
        self.instruction = "You are a professional simultaneous interpreter, your task is to translate the following {src_lang} text into {tgt_lang} with {latency} latency."
        self.latency = self.args.latency

        self.predictions = []

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

    def load_tokenizer_and_model(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            padding_side="right",
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        self.model.cuda()
        self.model.eval()

        self.template = TEMPLATES.get(self.args.template, None)
        _add_or_replace_eos_token(self.tokenizer, eos_token=self.template.stop_words[0])

    def prepare_gen_kwargs(self, args):
        gen_kwargs = {}
        gen_kwargs["do_sample"] = False
        gen_kwargs["temperature"]=None
        gen_kwargs["top_p"]=None
        gen_kwargs["max_new_tokens"] = args.max_new_tokens
        gen_kwargs["num_beams"] = args.num_beams
        gen_kwargs["eos_token_id"] = list(set([self.tokenizer.eos_token_id]))
        gen_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

        if self.args.template == "llama3":
            gen_kwargs["eos_token_id"].append(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
            gen_kwargs["eos_token_id"] = list(set(gen_kwargs["eos_token_id"]))
        
        return gen_kwargs

    def prepare_read_kwargs(self):
        self.gen_kwargs["suppress_tokens"] = self.read_suppress_tok_ids
        self.gen_kwargs["eos_token_id"] = self.read_eos_tok_ids
        self.gen_kwargs["num_beams"] = 1
        self.gen_kwargs["max_new_tokens"] = 1

    def prepare_write_kwargs(self, read_tok_num):
        self.gen_kwargs["num_beams"] = self.args.num_beams
        self.gen_kwargs["suppress_tokens"] = self.write_suppress_tok_ids
        self.gen_kwargs["eos_token_id"] = self.write_eos_tok_ids
        max_new_tokens = (read_tok_num + 25) * 2
        self.gen_kwargs["max_new_tokens"] = min(self.args.max_new_tokens, max_new_tokens)

    def set_special_tokens(self):
        self.eor_token = "<|end-of-read|>"
        self.eow_token = "<|end-of-write|>"

        eor_tok_id = self.tokenizer(self.eor_token, add_special_tokens=False).input_ids
        eow_tok_id = self.tokenizer(self.eow_token, add_special_tokens=False).input_ids

        self.eos_token = self.tokenizer.decode(self.tokenizer.eos_token_id)
        bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id == list else [self.tokenizer.bos_token_id]

        self.read_eos_tok_ids = eor_tok_id
        self.write_eos_tok_ids = self.gen_kwargs["eos_token_id"] + eow_tok_id

        self.read_suppress_tok_ids = self.gen_kwargs["eos_token_id"] + eow_tok_id + bos_token_id
        self.write_suppress_tok_ids = eor_tok_id + bos_token_id

    def load_eval_datasets(self, data_path):
        with open(data_path, "r") as f:
            data = json.load(f)
        logger.info("Loading dataset {}...".format(data_path))
        return data

    def eval_instance_with_beam_search(self, index, sample):
        src_text =  sample["source"]
        ref =  sample["reference"]
        src_lang =  sample["src_lang"]
        tgt_lang =  sample["tgt_lang"]

        instruction = self.instruction.format(src_lang=src_lang, tgt_lang=tgt_lang, latency=self.latency)
        messages = [{"role": "user", "content": instruction}]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        past_key_values = None
        prev_input_len = 0

        if src_lang == "Chinese":
            src_tokens = tokenize_chinese(src_text, self.tokenizer)
        else:
            src_tokens = src_text.split()

        input_text = prompt
        is_read = True

        idx = 0
        read_tok_num = 0
        preds = []
        read_chunk = []
        read_contents = []

        while idx < len(src_tokens) or (not is_read):
            prev_key_values = past_key_values
            if is_read:
                input_token = src_tokens[idx]
                if idx ==0 or src_lang == "Chinese":
                    input_text = f"{input_text}{input_token}"
                else:
                    input_text = f"{input_text} {input_token}"

                self.prepare_read_kwargs()
                idx += 1
                read_tok_num += 1
                read_chunk.append(input_token)
            else:
                if src_lang == "Chinese":
                    read_contents.append("".join(read_chunk))
                else:
                    read_contents.append(" ".join(read_chunk))
                
                read_chunk = []
                self.prepare_write_kwargs(read_tok_num)
                num_beams = self.gen_kwargs["num_beams"]

                if past_key_values is not None:
                    past_key_values = tuple((k.repeat(num_beams, 1, 1, 1), v.repeat(num_beams, 1, 1, 1)) for k, v in past_key_values)

            model_inputs = self.tokenizer([input_text], add_special_tokens=False, return_tensors="pt").to(self.device)

            curr_input_len = model_inputs.input_ids[0].size(0)

            if curr_input_len - prev_input_len < 1:
                less_token = 1 - (curr_input_len - prev_input_len)
                past_key_values = tuple((k[:,:,:-less_token,:], v[:,:,:-less_token,:]) for k, v in past_key_values)
            
            prev_input_len = curr_input_len

            model_output = self.model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                output_scores=True,
                return_dict_in_generate=True,
                past_key_values=past_key_values,
                **self.gen_kwargs
            )

            generated_ids = model_output.sequences
            past_key_values = model_output.past_key_values

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

            if not is_read:
                past_key_values = prev_key_values

            response = response.replace(self.eos_token, self.eow_token)
            
            # read mode and predicted token is "<|end-of-read|>"
            if is_read and self.eor_token in response:
                # switch write mode
                is_read = False
                input_text = f"{input_text}{response}"
            # write mode
            elif not is_read and (self.eow_token in response or generated_ids[0].size(0) >= self.gen_kwargs["max_new_tokens"]):
                is_read = True
                read_tok_num = 0
                hypo = response.rstrip().replace(self.eow_token, "")
                # write until `max_new_tokens` is reached
                if self.eow_token not in response:
                    response = f"{response}{self.eow_token}"
                input_text = f"{input_text}{response}"
                preds.append(hypo)
            # finish read
            elif idx >= len(src_tokens):
                is_read = False
                input_text = f"{input_text}{self.eor_token}"
                past_key_values = prev_key_values
            elif is_read and len(generated_ids[0]) > 1:
                past_key_values = prev_key_values
            elif is_read and self.args.document_level and input_text[-1] in "。？！.!?" and read_tok_num >= 20:
                is_read = False
                input_text = f"{input_text}{self.eor_token}"

        output = input_text[len(prompt):].strip()
        translation = "".join(preds)

        self.predictions.append(
            {
                "index": index,
                "source": src_text,
                "reference": ref,
                "prediction": translation,
                "output": output,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "read_contents": read_contents,
                "hypo": preds,
            }
        )

    def eval_instance_with_greedy_search(self, index, sample):
        src_text =  sample["source"]
        ref =  sample["reference"]
        src_lang =  sample["src_lang"]
        tgt_lang =  sample["tgt_lang"]

        instruction = self.instruction.format(src_lang=src_lang, tgt_lang=tgt_lang, latency=self.latency)
        messages = [{"role": "user", "content": instruction}]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        past_key_values = None

        if src_lang == "Chinese":
            src_tokens = tokenize_chinese(src_text, self.tokenizer)
        else:
            src_tokens = src_text.split()

        input_text = prompt
        is_read = True

        idx = 0
        preds = []
        read_tok_num = 0
        read_contents = []
        read_chunk = []

        while idx < len(src_tokens) or (not is_read):
            if is_read:
                input_token = src_tokens[idx]
                if idx ==0 or src_lang == "Chinese":
                    input_text = f"{input_text}{input_token}"
                else:
                    input_text = f"{input_text} {input_token}"

                self.prepare_read_kwargs()
                idx += 1
                read_tok_num += 1
                read_chunk.append(input_token)
            else:
                if src_lang == "Chinese":
                    read_contents.append("".join(read_chunk))
                else:
                    read_contents.append(" ".join(read_chunk))
                
                read_chunk = []
                self.prepare_write_kwargs(read_tok_num)

            model_inputs = self.tokenizer([input_text], add_special_tokens=False, return_tensors="pt").to(self.device)

            model_output = self.model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                output_scores=True,
                return_dict_in_generate=True,
                past_key_values=past_key_values,
                **self.gen_kwargs
            )

            generated_ids = model_output.sequences
            past_key_values = model_output.past_key_values

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

            response = response.replace(self.eos_token, self.eow_token)
            
            # read mode and predicted token is "<|end-of-read|>"
            if is_read and self.eor_token in response:
                # switch to writing mode
                is_read = False
                input_text = f"{input_text}{response}"
            # write mode
            elif not is_read and (self.eow_token in response or generated_ids[0].size(0) >= self.gen_kwargs["max_new_tokens"]):
                hypo = response.rstrip().replace(self.eow_token, "")
                # write until `max_new_tokens` is reached
                if self.eow_token not in response:
                    response = f"{response}{self.eow_token}"
                input_text = f"{input_text}{response}"
                preds.append(hypo)
                # swith to reading mode
                is_read = True
                read_tok_num = 0
            # finish read
            elif idx >= len(src_tokens):
                is_read = False
                input_text = f"{input_text}{self.eor_token}"
            elif is_read and self.args.document_level and input_text[-1] in "。？！.!?" and read_tok_num >= 20:
                is_read = False
                input_text = f"{input_text}{self.eor_token}"

        output = input_text[len(prompt):].strip()
        translation = "".join(preds)

        self.predictions.append(
            {
                "index": index,
                "source": src_text,
                "reference": ref,
                "prediction": translation,
                "output": output,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "read_contents": read_contents,
                "hypo": preds,
            }
        )

    def cal_scores(self):

        hypos = []
        refs = []
        results = {}
        ALs = []
        LAALs = []

        for prediction in self.predictions:
            translation = prediction["prediction"]
            ref = prediction["reference"]
            src_lang = prediction["src_lang"]
            output = prediction["output"]
            tgt_lang = prediction["tgt_lang"]
            read_contents = prediction["read_contents"]
            hypo = prediction["hypo"]

            hypos.append(translation)
            refs.append(ref)

            if tgt_lang == "Chinese":
                tok = "zh"
                pred_len = len(list(translation))
                ref_len = len(list(ref))
            else:
                tok = "13a"
                pred_len = len(translation.split())
                ref_len = len(ref.split())

            if src_lang == "Chinese":
                src_len = len(list(prediction["source"]))
            else:
                src_len = len(prediction["source"].split())

            bleu = sacrebleu.sentence_bleu(translation, [ref], tokenize=tok).score

            delays, src_len_ = compute_delays(read_contents, hypo, src_lang, tgt_lang)
            AL = AverageLagging(delays, src_len, ref_len)
            LAAL = LengthAdaptiveAverageLagging(delays, src_len, ref_len)

            ALs.append(AL)
            LAALs.append(LAAL)
            
            prediction["delays"] = str(delays)
            prediction["BLEU"] = bleu
            prediction["AL"] = AL
            prediction["LAAL"] = LAAL

        if self.predictions[0]["tgt_lang"] == "Chinese":
            tok = "zh"
        else:
            tok = "13a"

        bleu_score = sacrebleu.corpus_bleu(
                        hypos,
                        [refs],
                        tokenize=tok
                    ).score

        comet_score, bleurt_score = self.compute_comet_bleurt()

        results["BLEU"] = bleu_score
        results["COMET"] = comet_score
        results["BLEURT"] = bleurt_score
        results["AL"] = mean(ALs)
        results["LAAL"] = mean(LAALs)
    
        return results

    def compute_comet_bleurt(self):
        # COMET
        comet_model = load_from_checkpoint(self.args.comet_ckpt_path, reload_hparams=True)

        data = [
            {
                "src": item["source"],
                "mt": item["prediction"],
                "ref": item["reference"]
            }
            for item in self.predictions
        ]

        comet_output = comet_model.predict(data, batch_size=256, gpus=1)
        for idx, comet_score in enumerate(comet_output.scores):
            self.predictions[idx]["COMET"] = comet_score * 100

        COMET = comet_output.system_score * 100

        torch.cuda.empty_cache()

        # BLEURT
        scorer = score.BleurtScorer(self.args.bleurt_ckpt_path)
        bleurt_scores = []

        for prediction in self.predictions:
            ref = prediction["reference"]
            hypo = prediction["prediction"]
            bleurt = scorer.score(references=[ref], candidates=[hypo])[0] * 100
            prediction["BLEURT"] = bleurt
            bleurt_scores.append(bleurt)

        return COMET, mean(bleurt_scores)


    def save_results(self, results, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        prediction_file = os.path.join(output_dir, "prediction.json")
        with open(prediction_file, "w", encoding="utf8") as f:
            json.dump(self.predictions, f, ensure_ascii=False, indent=4)

        result_file = os.path.join(output_dir, "results.json")
        with open(result_file, "w", encoding="utf8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    def simul_eval(self):
        if self.args.num_beams == 1:
            eval_instance_func = self.eval_instance_with_greedy_search
        elif self.args.num_beams > 1:
            eval_instance_func = self.eval_instance_with_beam_search
        else:
            raise ValueError("num_beams must be greater then or equal to 1.")

        with torch.no_grad():
            for index, sample in tqdm(enumerate(self.test_data), total=len(self.test_data)):
                eval_instance_func(index, sample)
        
        self.model.cpu()
        torch.cuda.empty_cache()

        results = self.cal_scores()

        self.save_results(results, self.args.output_dir)


def load_infer_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path to the model file", required=True
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Path to the input file", required=True
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Path to the output file", required=True
    )
    parser.add_argument(
        "--num_beams", type=int, default=5,
        help="number of beams of beam search"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=128,
        help="max new tokens for generation"
    )
    parser.add_argument(
        "--template", type=str, default=None,
        help="Which template to use for constructing prompts in training and inference.", required=True
    )
    parser.add_argument(
        "--latency", type=str, default=None,
        help="Latency level for streaming inference", required=True
    )
    parser.add_argument(
        "--bleurt_ckpt_path", type=str, default=None,
        help="Path to BLEURT checkpoint", required=True
    )
    parser.add_argument(
        "--comet_ckpt_path", type=str, default=None,
        help="Path to COMET checkpoint", required=True
    )
    parser.add_argument(
        "--document_level", type=bool, default=False,
        help="document level infer", required=False
    )
    args = parser.parse_args()

    return args


def run_simuleval():
    args = load_infer_args()

    infer = SimulInference(args)
    infer.simul_eval()


if __name__ == "__main__":
    run_simuleval()