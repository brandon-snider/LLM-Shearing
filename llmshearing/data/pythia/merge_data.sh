PROJ_DIR=/Users/brandon/Documents/College/q4-fall-24/cs-229/project/LLM-Shearing

python ${PROJ_DIR}/llmshearing/data/pythia/merge_data.py \
    --input_dir=${PROJ_DIR}/data/mds/eval \
    --output_dir=${PROJ_DIR}/data/mds/for_prune/eval_merge \
    --output_split=eval_merge \
    --split_names arxiv book c4-rp cc github stackexchange wiki \
    --shuffle
