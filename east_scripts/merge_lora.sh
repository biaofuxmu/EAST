
BASE_MODEL_PATH=/path/to/stage1_model

LORA_MODEL_PATH=/path/to/stage2_lora_model

OUTPUT_DIR=${LORA_MODEL_PATH}/merged_lora_model

template=llama3

llamafactory-cli export \
    --model_name_or_path ${BASE_MODEL_PATH} \
    --adapter_name_or_path ${LORA_MODEL_PATH} \
    --template ${template} \
    --finetuning_type lora \
    --export_dir ${OUTPUT_DIR} \
    --export_size 4 \
    --export_device cpu \
    --export_legacy_format false
