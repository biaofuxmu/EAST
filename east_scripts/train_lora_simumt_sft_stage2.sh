
dataset="simt_multi_90k,offline_multi_120k"
BASE_MODEL_PATH=./save_model/EAST-Stage-I-8B
SAVE_PATH=./save_models/EAST-Stage2-Lora


llamafactory-cli train \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --ddp_timeout 180000000 \
    --flash_attn fa2 \
    \
    --model_name_or_path ${BASE_MODEL_PATH} \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --lora_target all \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
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
    --num_train_epochs 2 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 True \
