PROJ_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd ../../../ && pwd )"

python ${PROJ_DIR}/llmshearing/data/opencoder/merge_data.py \
    --input_dir=${PROJ_DIR}/data/opencoder-annealing/pythia/for_prune/eval \
    --output_dir=${PROJ_DIR}/data/opencoder-annealing/pythia/for_prune/eval/merged \
    --output_split=eval_merged \
    --split_names algorithmic_corpus synthetic_code_snippet synthetic_qa \
    --shuffle
