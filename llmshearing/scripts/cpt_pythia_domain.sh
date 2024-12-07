PROJ_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd ../../ && pwd )"

TRAIN_SCRIPT=${PROJ_DIR}/llmshearing/train.py
DATA_DIR=${PROJ_DIR}/data/for_prune/code_5b/pythia/mds
OUTPUT_DIR=${PROJ_DIR}/out/cpt_pythia_domain
LAUNCH_SCRIPT=${PROJ_DIR}/llmshearing/scripts/launch.sh

model=160m
config_file=${PROJ_DIR}/llmshearing/configs/pythia/${model}.yaml

# Path to the pruned model
path=${PROJ_DIR}/out/prune_pythia_domain/pythia_410m__160m_2048_code_5b/pruned-latest-rank0.pt

# basic setup
max_seq_len=2048
device_train_microbatch_size=16
global_train_batch_size=256
device_eval_batch_size=32
n_gpus=1

# learning setup
lr=1e-4 # learning rate for the main parameters
max_duration=10000ba # 5B tokens
save_interval=500ba # save every 3200ba
t_warmup=300ba # 3% learning rate warmup 

# dynamic loading setup
# dynamic=True
# set_names=[cc,github,book,stackexchange,wiki,arxiv,c4-rp] # domain names
# proportion=[0.2192,0.0002,0.0791,0.0064,0.0096,0.001,0.6845] # final proportion of pruning

dynamic=False
set_names=[train]
proportion=[1.0]

dataset_name=code_5b
train_split_name=train
eval_split_name=eval # eval on all domains
eval_target_model=false # evaluate on the current model, not the target model, otherwise the loss will be inaccurate
# eval_subset_num_batches=440 # should be 3,500 / device_eval_batch_size (I think)
eval_interval=500ba # eval at this interval

# save directroy
run_name=pythia_${model}_cpt_${max_duration}_${dataset_name}
save_dir=${OUTPUT_DIR}/${run_name}
wandb_dir=${save_dir} # save locally
     

# doremi: update weights with exponential descent
# constant: keep the weights constant
update_type=doremi 
if [[ $model == 6.6m ]]; then
    target_loss=[4.26942,2.17134,4.2438,3.14441,4.45089,2.96644,4.52761] # Loss on eval set of 14m model
elif [[ $model == 14m ]]; then
    target_loss=[4.26942,2.17134,4.2438,3.14441,4.45089,2.96644,4.52761] # Loss on eval set of 14m model
elif [[ $model == 70m ]]; then
    target_loss=[3.7480,1.6161,3.6199,2.6632,3.6930,2.4198,3.9704] # Loss on eval set of pretrained 70m model
elif [[ $model == 160m ]]; then
    target_loss=[3.3283,1.3144,3.2101,2.3087,3.2770,2.0879,3.5596] # Loss on eval set of pretrained 160m model
elif [[ $model == 410m ]]; then
    target_loss=[2.8459,1.0265,2.7563,1.9353,2.6475,1.7337,3.0853] # Loss on eval set of pretrained 410m model
fi


# Run in bash, it will automatically use resources available in the current environment
composer \
    -n ${n_gpus} \
    $TRAIN_SCRIPT \
    $config_file \
    run_name=${run_name} \
    data_local=${DATA_DIR} \
    eval_loader.dataset.split=${eval_split_name} \
    global_train_batch_size=${global_train_batch_size} \
    device_train_microbatch_size=${device_train_microbatch_size} \
    device_eval_batch_size=${device_eval_batch_size} \
    max_seq_len=${max_seq_len} \
    max_duration=${max_duration} \
    eval_first=true \
    scheduler.t_warmup=${t_warmup} \
    save_folder=${save_dir} \
    loggers.wandb.init_kwargs.dir=${wandb_dir} \
    eval_interval=${eval_interval} \
    save_interval=${save_interval} \
    optimizer.lr=${lr} \
    model.l0_module=null \
    model.path=${path} \
    callbacks.data_loading.dynamic=${dynamic} \
    callbacks.data_loading.set_names=${set_names} \
    callbacks.data_loading.proportion=${proportion} \
    callbacks.data_loading.update_type=${update_type} \
    callbacks.data_loading.target_loss=${target_loss} \
    train_loader.num_workers=0 \
    train_loader.prefetch_factor=null \
    train_loader.persistent_workers=false \
    train_loader.dataset.split=${train_split_name} \
    autoresume=true