PROJ_DIR=$1

python ${PROJ_DIR}/llmshearing/data/pythia/merge_data.py \
    --input_dir=${PROJ_DIR}/data/redpajama-1B/pythia/mds/for_prune \
    --output_dir=${PROJ_DIR}/data/redpajama-1B/pythia/mds/merged \
    --output_split=trains \
    --split_names arxiv book c4-rp cc github stackexchange wiki \
    --shuffle
