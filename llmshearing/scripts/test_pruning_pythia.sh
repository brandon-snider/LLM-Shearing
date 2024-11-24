# pruning pythia 14m -> 7m 

# Please specify the working folder
PROJ_DIR=/Users/brandon/Documents/College/q4-fall-24/cs-229/project/LLM-Shearing
LAUNCH_SCRIPT=${PROJ_DIR}/llmshearing/scripts/launch.sh
DATA_DIR=${PROJ_DIR}/data/mds/for_prune
OUTPUT_DIR=${PROJ_DIR}/out/test_pruning_pythia
TRAIN_SCRIPT=${PROJ_DIR}/llmshearing/train.py
MODEL_PATH=${PROJ_DIR}/models/pythia-14m-composer

# Specify $PROJ_DIR in scripts/launch.sh and scripts/srun_launch.sh if using slurm

test=False

from_model=14m # source model size
to_model=6.6m # target model size
config_file=${PROJ_DIR}/llmshearing/configs/pythia/${from_model}.yaml
path=$MODEL_PATH/state_dict.pt

# data setup
data_local=${DATA_DIR}

# basic setup
max_seq_len=2048
device_train_microbatch_size=1
global_train_batch_size=4
device_eval_batch_size=1

# learning setup
lr=3e-4 # learning rate for the main parameters
max_duration=10ba # 82k tokens
save_interval=10ba # save in the end
t_warmup=10ba # learning rate warmup (typically set to 10% of max_duration)

# dynamic loading setup
dynamic=True
set_names=[cc,github,book,stackexchange,wiki,arxiv,c4-rp] # domain names
proportion=[0.67,0.045,0.045,0.02,0.045,0.025,0.15] # initial proportion of RP, make sure that the sum(proportion) = 1

# doremi: update weights with exponential descent
# constant: keep the weights constant
update_type=doremi 
if [[ $to_model == 6.6m ]]; then
    target_loss=[4.26942,2.17134,4.2438,3.14441,4.45089,2.96644,4.52761] # Loss on eval set of 14m model
fi
eval_split_name=eval_merge # eval on all domains
eval_target_model=false # evaluate on the current model, not the target model, otherwise the loss will be inaccurate
eval_interval=1ba # eval at this interval and update the loading proportion


# pruning setup
lag_lr=1.0 # learning rate or l0_module
lagr_warmup=10ba # sparsity warmup (typically set to 20% of max_duration)
if [[ $to_model == 6.6m ]]; then
    target_d_model=64; target_n_heads=3; target_n_layers=4; target_intermediate_size=256
fi

# save directroy
run_name=pythia_${from_model}_pruning_scaling_${update_type}_to${to_model}_sl${max_seq_len}
save_dir=${OUTPUT_DIR}/${run_name}
wandb_dir=${save_dir} # save locally

if [[ $test == True ]]; then t=00-01:00:00; else t=01-00:00:00; fi

# Run in bash, it will automatically use resources available in the current environment
# composer $TRAIN_SCRIPT \

# Run with slurm    
# sbatch --job-name ${run_name} \
#     --nodes=2 \
#     --gpus-per-node=4 \
#     --mem=512gb \
#     --cpus-per-task=8 \
#     --time $t \
#     -p pli \
    # $LAUNCH_SCRIPT \
    composer \
    -n 1 \
    $TRAIN_SCRIPT \
    $config_file \
    run_name=${run_name} \
    data_local=${data_local} \
    eval_loader.dataset.split=${eval_split_name} \
    global_train_batch_size=${global_train_batch_size} \
    device_train_microbatch_size=${device_train_microbatch_size} \
    device_eval_batch_size=${device_eval_batch_size} \
    max_seq_len=${max_seq_len} \
    max_duration=${max_duration} \
    eval_first=false \
    scheduler.t_warmup=${t_warmup} \
    save_folder=${save_dir} \
    loggers.wandb.init_kwargs.dir=${wandb_dir} \
    eval_interval=${eval_interval} \
    save_interval=${save_interval} \
    optimizer.lr=${lr} \
    optimizer.lag_lr=${lag_lr} \
    model.path=${path} \
    model.l0_module.lagrangian_warmup_steps=${lagr_warmup} \
    model.l0_module.pruning_modules='[head,intermediate,layer,hidden]' \
    model.l0_module.eval_target_model=${eval_target_model} \
    model.l0_module.target_model.d_model=${target_d_model} \
    model.l0_module.target_model.n_heads=${target_n_heads} \
    model.l0_module.target_model.n_layers=${target_n_layers} \
    model.l0_module.target_model.intermediate_size=${target_intermediate_size} \
    callbacks.data_loading.dynamic=${dynamic} \
    callbacks.data_loading.set_names=${set_names} \
    callbacks.data_loading.proportion=${proportion} \
    callbacks.data_loading.update_type=${update_type} \
    callbacks.data_loading.target_loss=${target_loss} \
    train_loader.num_workers=0 \
    train_loader.prefetch_factor=None \
    train_loader.persistent_workers=false \
    autoresume=false