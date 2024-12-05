PROJ_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd ../../ && pwd )"

TRAIN_SCRIPT=${PROJ_DIR}/llmshearing/train.py
DATA_DIR=${PROJ_DIR}/data/opencoder-annealing/pythia/for_prune_merged
OUTPUT_DIR=${PROJ_DIR}/out/prune_pythia
MODEL_PATH=${PROJ_DIR}/models/pythia-160m-composer # Used to load the pretrianed weights

from_model=160m # source model size
to_model=70m # target model size
config_file=${PROJ_DIR}/llmshearing/configs/pythia/${from_model}.yaml
path=$MODEL_PATH/state_dict.pt

# basic setup
max_seq_len=2048
device_train_microbatch_size=32
global_train_batch_size=32
device_eval_batch_size=64
n_gpus=1

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
# eval_subset_num_batches=440 # should be 3,500 / device_eval_batch_size (I think)
eval_interval=200ba # eval at this interval

# pruning setup
lag_lr=1.0 # learning rate or l0_module
lagr_warmup=640ba # sparsity warmup (typically set to 20% of max_duration)

# save directroy
run_name=pythia_${from_model}_${update_type}_${to_model}_${max_seq_len}_${dataset_name}
save_dir=${OUTPUT_DIR}/${run_name}
wandb_dir=${save_dir}

# doremi: update weights with exponential descent
# constant: keep the weights constant
update_type=doremi 

if [[ $to_model == 6.6m ]]; then
    target_loss=[4.26942,2.17134,4.2438,3.14441,4.45089,2.96644,4.52761] # Loss on eval set of 14m model
elif [[ $to_model == 14m ]]; then
    target_loss=[4.26942,2.17134,4.2438,3.14441,4.45089,2.96644,4.52761] # Loss on eval set of 14m model
elif [[ $to_model == 70m ]]; then
    target_loss=[3.7480,1.6161,3.6199,2.6632,3.6930,2.4198,3.9704] # Loss on eval set of pretrained 70m model
elif [[ $to_model == 160m ]]; then
    target_loss=[3.3283,1.3144,3.2101,2.3087,3.2770,2.0879,3.5596] # Loss on eval set of pretrained 160m model
elif [[ $to_model == 410m ]]; then
    target_loss=[2.8459,1.0265,2.7563,1.9353,2.6475,1.7337,3.0853] # Loss on eval set of pretrained 410m model
fi

if [[ $to_model == 6.6m ]]; then
    target_d_model=64; target_n_heads=4; target_n_layers=4; target_intermediate_size=256
elif [[ $to_model == 14m ]]; then
    target_d_model=128; target_n_heads=4; target_n_layers=6; target_intermediate_size=512
elif [[ $to_model == 70m ]]; then
    target_d_model=512; target_n_heads=6; target_n_layers=8; target_intermediate_size=2048
elif [[ $to_model == 160m ]]; then
    target_d_model=768; target_n_heads=12; target_n_layers=12; target_intermediate_size=3072
elif [[ $to_model == 410m ]]; then
    target_d_model=1024; target_n_heads=24; target_n_layers=16; target_intermediate_size=4096
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
