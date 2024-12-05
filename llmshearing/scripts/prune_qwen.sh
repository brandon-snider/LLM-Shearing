PROJ_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd ../../ && pwd )"
TRAIN_SCRIPT=${PROJ_DIR}/llmshearing/train.py
DATA_DIR=${PROJ_DIR}/data/opencoder-annealing/qwen/for_prune_merged
OUTPUT_DIR=${PROJ_DIR}/out/prune_qwen_opencoder
MODEL_PATH=${PROJ_DIR}/models/qwen-1.5b-composer # Used to load the pretrianed weights

from_model=1.5b # source model size
to_model=0.5b # target model size
config_file=${PROJ_DIR}/llmshearing/configs/qwen/${from_model}.yaml
path=$MODEL_PATH/state_dict.pt
tokenizer_path=${PROJ_DIR}/tokenizers/qwen-tokenizer

# basic setup
max_seq_len=2048
device_train_microbatch_size=8
global_train_batch_size=32
device_eval_batch_size=16
n_gpus=4

# learning setup
lr=1e-4 # learning rate for the main parameters
max_duration=6400ba # 400M tokens
save_interval=400ba # save in the end
t_warmup=320ba # learning rate warmup (typically set to 10% of max_duration)

# dynamic=True
# set_names=[cc,github,book,stackexchange,wiki,arxiv,c4-rp] # domain names
# proportion=[0.67,0.045,0.045,0.02,0.045,0.025,0.15] # initial proportion of RP, make sure that the sum(proportion) = 1

dynamic=False
set_names=[train]
proportion=[1.0]

dataset_name=opencoder-annealing
train_split_name=train
eval_split_name=eval # eval on all domains
eval_target_model=false # evaluate on the current model, not the target model, otherwise the loss will be inaccurate
eval_subset_num_batches=440 # should be 3,500 / device_eval_batch_size (I think)
eval_interval=200ba # eval at this interval

# pruning setup
lag_lr=1.0 # learning rate or l0_module
lagr_warmup=640ba # sparsity warmup (typically set to 20% of max_duration)

# save directroy
run_name=qwen_${from_model}_${update_type}_${to_model}_${max_seq_len}_${dataset_name}
save_dir=${OUTPUT_DIR}/${run_name}
wandb_dir=${save_dir}

# doremi: update weights with exponential descent
# constant: keep the weights constant
update_type=doremi 

if [[ $to_model == 0.5b ]]; then
    target_loss=[4.26942,2.17134,4.2438,3.14441,4.45089,2.96644,4.52761] # Note: only used with dynamic batch loading
fi

if [[ $to_model == 0.5b ]]; then
    target_d_model=896; target_n_heads=12; target_n_layers=24; target_intermediate_size=4864
fi

composer \
    -n ${n_gpus} \
    $TRAIN_SCRIPT \
    $config_file \
    n_devices=${n_gpus} \
    run_name=${run_name} \
    data_local=${DATA_DIR} \
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
    eval_subset_num_batches=${eval_subset_num_batches} \
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
    train_loader.prefetch_factor=null \
    train_loader.persistent_workers=false \
    train_loader.dataset.split=${train_split_name} \
    autoresume=false
