import json

data_paths = [
    "/path/to/SiMT-De-En-660K.json",
    "/path/to/SiMT-Multi-90K.json"
]

output_paths = [
    "./data/mt_data/train_data/SiMT-De-En-660K.json",
    "./data/mt_data/train_data/SiMT-Multi-90K.json"
]

for data_path, output_path in zip(data_paths, output_paths):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []

    for idx, item in enumerate(data):

        src_lang = item["src_lang"]
        tgt_lang = item["tgt_lang"]
        latency = item["latency"]
        source_chunks = item["source_chunks"]
        target_chunks = item["target_chunks"]

        src_i = 0

        output = ""
        for src, tgt in zip(source_chunks, target_chunks):
            src = src.strip()
            tgt = tgt.strip()

            src_seg = "" if src_lang == "Chinese" else " "
            tgt_seg = "" if tgt_lang == "Chinese" else " "

            if src_i > 0:
                src = f"{src_seg}{src}"
                tgt = f"{tgt_seg}{tgt}"

            output = output + f"{src}<|end-of-read|>{tgt}<|end-of-write|>"

        samples.append(
            {
                "instruction": f"You are a professional simultaneous interpreter, your task is to translate the following {src_lang} text into {tgt_lang} with {latency} latency.",
                "input": "", 
                "output": output
            }
        )

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=4)