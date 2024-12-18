NUM_DOMAINS=7
echo "Total domains: $NUM_DOMAINS"

PROJ_DIR=$1

for ((i=0; i<NUM_DOMAINS; i++))
do
    echo $i
    SLURM_ARRAY_TASK_ID=$i python ${PROJ_DIR}/llmshearing/data/pythia/sample.py \
        --target_dir ${PROJ_DIR}/data/mds \
        --tokenized_dir ${PROJ_DIR}/data/split \
        --eval_seq 500
done
