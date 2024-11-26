model=$1
modelname=$(basename "$model")

# header for running slurm jobs for evaluation
# sbatch --output=$harness_dir/slurm/%A-%x.out -N 1 -n 1 --mem=200G --cpus-per-task 10  --gres=gpu:a100:1 --mail-type=FAIL,TIME_LIMIT --mail-user=mengzhou@cs.princeton.edu --time 1:00:00 --job-name harnesspythia-$modelname -x "della-i14g[1-20]" <<EOF
# #!/bin/bash
# EOF

harness_dir="$(pwd)"
output_path="${modelname}_output.json"

# zero-shot evaluation for Pythia evaluation tasks

bash hf_open_llm.sh $model lambada_openai,piqa,winogrande,wsc,arc_challenge,arc_easy,sciq,logiqa 0 pythia0shot-$modelname


# five-shot evaluation for Pythia evaluation tasks
bash hf_open_llm.sh $model lambada_openai,piqa,winogrande,wsc,arc_challenge,arc_easy,sciq,logiqa 5 pythia5shot-$modelname


# HF leaderboard evaluation
bash $harness_dir/hf_open_llm.sh $model hendrycks_math,hendrycks_math_algebra,hendrycks_math_counting_and_prob,hendrycks_math_geometry,hendrycks_math_intermediate_algebra,hendrycks_math_num_theory,hendrycks_math_prealgebra,hendrycks_math_precalc 5 mmlu5shot-$modelname
bash $harness_dir/hf_open_llm.sh $model hellaswag 10 hellaswag5shot-$modelname
bash $harness_dir/hf_open_llm.sh $model arc_challenge 25 arcc5shot-$modelname
bash $harness_dir/hf_open_llm.sh $model truthfulqa_ml_mc1 0 truthfulqa5shot-$modelname
bash $harness_dir/hf_open_llm.sh $model truthfulqa_mc 0 truthfulqa5shot-$modelname

# others
bash $harness_dir/hf_open_llm.sh $model nq_open 32 nq_open32shot-$modelname
bash $harness_dir/hf_open_llm.sh $model boolq 32 boolq32shot-$modelname
bash $harness_dir/hf_open_llm.sh $model gsm8k  8 gsm8k8shot-$modelname