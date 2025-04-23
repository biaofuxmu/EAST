
SAVE_PATH=/path/to/EAST-8B

template=llama3

lang_pair=de-en
year=22
beam=5

for latency in "low" "low-medium" "medium" "medium-high" "high"
do
    dataset=./data/mt_data/test_data/wmt${year}.test.${lang_pair}.json
    RESULT_PATH=${SAVE_PATH}/simul_results/predict_wmt${year}_test_${lang_pair}_${latency}_beam${beam}

    echo ${template}
    echo ${dataset}
    echo ${RESULT_PATH}

    python ./user_scripts/eval/simuleval.py \
        --model_path ${SAVE_PATH} \
        --data_path ${dataset} \
        --output_dir ${RESULT_PATH} \
        --template ${template} \
        --max_new_tokens 1024 \
        --latency ${latency} \
        --num_beams ${beam} \
        --bleurt_ckpt_path "/path/to/BLEURT-20" \
        --comet_ckpt_path "/path/to/wmt22-comet-da/checkpoints/model.ckpt"
done
