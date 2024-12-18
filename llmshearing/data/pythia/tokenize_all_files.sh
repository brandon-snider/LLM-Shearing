NUM_FILES=$(wc -l < jsonl_list.txt)
echo "Total files: $NUM_FILES"

PROJ_DIR=$1

for ((i=0; i<NUM_FILES; i++))
do
    echo $i
    SLURM_ARRAY_TASK_ID=$i python ${PROJ_DIR}/llmshearing/data/pythia/tokenize_single_file.py \
        --target_dir ${PROJ_DIR}/data/tokenized \
        --raw_dir ${PROJ_DIR}/data/raw \
        --seq_length 2048
done
