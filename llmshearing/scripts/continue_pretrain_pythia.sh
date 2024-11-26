# pruning llama2 7b -> 3b or 1.3b

PROJ_DIR=$1
DATA_DIR=${PROJ_DIR}/data/redpajama-1B/pythia/mds/for_prune
OUTPUT_DIR=${PROJ_DIR}/out/test_cpt_pythia
LAUNCH_SCRIPT=${PROJ_DIR}/llmshearing/scripts/launch.sh
TRAIN_SCRIPT=${PROJ_DIR}/llmshearing/train.py

test=True

model=160m
config_file=${PROJ_DIR}/llmshearing/configs/pythia/${model}.yaml

# Name of the pruning run that produced this model
prune_run_name=pythia_410m_unpruned

# Path to the pruned model
# path=${PROJ_DIR}/models/pythia-14m-composer/state_dict.pt
path=${PROJ_DIR}/out/test_pruning_pythia/pythia_410m_doremi_160m_2048/pruned-latest-rank0.pt

# data setup
data_local=${DATA_DIR}

# basic setup
max_seq_len=2048
device_train_microbatch_size=32
global_train_batch_size=256
device_eval_batch_size=64

# learning setup
lr=2e-4 # learning rate for the main parameters
max_duration=48000ba # 50B tokens
# max_duration=48000ba # 50B tokens
save_interval=400ba # save every 3200ba
# save_interval=3200ba # save every 3200ba
t_warmup=144ba # 3% learning rate warmup 

# dynamic loading setup
dynamic=True
set_names=[cc,github,book,stackexchange,wiki,arxiv,c4-rp] # domain names
proportion=[0.2192,0.0002,0.0791,0.0064,0.0096,0.001,0.6845] # final proportion of pruning
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
eval_split_name=eval_merge # eval on all domains
eval_interval=100ba # eval every 50 batches and update the loading proportion


# save directroy
# run_name=${prune_run_name}_ft${max_duration}
run_name=pythia_${model}_ft${max_duration}
save_dir=${OUTPUT_DIR}/${run_name}
wandb_dir=${save_dir} # save locally

if [[ $test == True ]]; then t=00-01:00:00; else t=01-00:00:00; fi

# Run with slurm
# sbatch -p cli \
#     --job-name ${run_name} \
#     --nodes=8 \
#     --gpus-per-node=2 \
#     --mem=512gb \
#     --cpus-per-task=8 \
#     --time $t \
#     $LAUNCH_SCRIPT \
     

# Run in bash, it will automatically use resources available in the current environment
composer \
    -n 2 \
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
    autoresume=false

# checking eval_first