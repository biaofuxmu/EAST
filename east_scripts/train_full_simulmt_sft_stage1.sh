
BASE_MODEL_PATH=/path/to/Llama-3-8B-Instruct

dataset="simt_de_en_660k"

SAVE_PATH=./save_model/EAST-Stage-I-8B

template=llama3

llamafactory-cli train \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --ddp_timeout 180000000 \
    --flash_attn fa2 \
    --resize_vocab \
    \
    --model_name_or_path ${BASE_MODEL_PATH} \
    --stage sft \
    --do_train \
    --finetuning_type full \
    \
    --dataset ${dataset} \
    --template llama3 \
    --cutoff_len 1024 \
    --overwrite_cache \
    --preprocessing_num_workers 32 \
    \
    --output_dir ${SAVE_PATH} \
    --logging_steps 10 \
    --save_steps 0.2 \
    --plot_loss true \
    --overwrite_output_dir \
    \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 True \
