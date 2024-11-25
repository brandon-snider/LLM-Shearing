PROJ_DIR=$1

python ${PROJ_DIR}/llmshearing/data/pythia/merge_data.py \
    --input_dir=${PROJ_DIR}/data/mds/eval \
    --output_dir=${PROJ_DIR}/data/mds/for_prune/eval_merge \
    --output_split=eval_merge \
    --split_names arxiv book c4-rp cc github stackexchange wiki \
    --shuffle
