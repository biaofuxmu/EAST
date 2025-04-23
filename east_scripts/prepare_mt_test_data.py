import json
import os

LANG_TABLE = {
    "en": "English",
    "de": "German",
    "cs": "Czech",
    "zh": "Chinese",
    "ru": "Russian",
}

lang_pairs = [ 'de-en', 'zh-en', 'ru-en', 'cs-en', 'en-de', 'en-zh', 'en-ru', 'en-cs'] 

for lang_pair in lang_pairs:
    slang = lang_pair.split("-")[0]
    tlang = lang_pair.split("-")[1]

    src_file = f'/path/to/wmt22.test.{lang_pair}.{slang}'
    tgt_file = f'/path/to/wmt22.test.{lang_pair}.{tlang}'
    output_file = f'data/mt_data/test_data/wmt22.test.{lang_pair}.json'

    src_lang = LANG_TABLE[slang]
    tgt_lang = LANG_TABLE[tlang]

    with open(src_file, 'r', encoding='utf-8') as f_src, open(tgt_file, 'r', encoding='utf-8') as f_tgt:
        source_sentences = f_src.readlines()
        target_sentences = f_tgt.readlines()

    assert len(source_sentences) == len(target_sentences)

    parallels = [
        {
            "source": f"{src.strip()}", 
            "reference": f"{tgt.strip()}",
            "src_lang": f"{src_lang}",
            "tgt_lang": f"{tgt_lang}",
        }
        for src, tgt in zip(source_sentences, target_sentences)
    ]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(parallels, f, ensure_ascii=False, indent=4)
