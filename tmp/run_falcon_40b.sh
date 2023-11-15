export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_PROCESS_LAUNCHER=none

export FI_PROVIDER=tcp
export CCL_ATL_TRANSPORT=mpi
export CCL_ATL_TRANSPORT=ofi
export CCL_ATL_SHM=1
export CCL_WORKER_COUNT=1
export DS_ACCELERATOR=xpu

# # 60 layer: ram rss:159Gi
export TASK_NAME=mrpc
deepspeed --num_gpus=12 ../transformers/examples/pytorch/text-classification/run_glue_no_trainer.py \
--model_name_or_path falcon-40b \
--task_name $TASK_NAME \
--max_length 128 \
--per_device_train_batch_size 1 \
--learning_rate 2e-5 \
--num_train_epochs 3 \
--output_dir log/Llama/$TASK_NAME/

# export TASK_NAME=mrpc
# deepspeed --num_gpus=12 ../transformers/examples/pytorch/text-classification/run_glue.py \
# --deepspeed ../transformers/tests/deepspeed/ds_config_zero2.json \
# --model_name_or_path falcon-40b \
# --task_name $TASK_NAME \
# --do_train \
# --do_eval \
# --max_seq_length 128 \
# --per_device_train_batch_size 1 \
# --learning_rate 2e-5 \
# --num_train_epochs 3 \
# --output_dir log/Llama/$TASK_NAME/ \
# --overwrite_output_dir \
# --use_ipex True



# bash run_falcon_40b.sh 2>&1 | tee log/fc/log_Falcon_40b_layer50_bs1_rank12_debug.log

# python -c 'from transformers import AutoModel; \
# from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
# model = AutoModel.from_pretrained("falcon-40b"); \
# estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=12, num_nodes=1)'
# Estimated memory needed for params, optim states and gradients for a:
# HW: Setup with 1 node, 12 GPUs per node.
# SW: Model with 41303M total params, 532M largest layer params.
#   per CPU  |  per GPU |   Options
#  1038.60GB |   1.98GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
#  2769.60GB |   1.98GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
#   923.20GB |   8.40GB | offload_param=none, offload_optimizer=cpu , zero_init=1
#  2769.60GB |   8.40GB | offload_param=none, offload_optimizer=cpu , zero_init=0
#    35.72GB |  59.68GB | offload_param=none, offload_optimizer=none, zero_init=1
#  2769.60GB |  59.68GB | offload_param=none, offload_optimizer=none, zero_init=0

# SYSMON:
# 40 layer: 56407.5 = x + 39  (no offload: 54871.8)  27.7B para
# 30 layer: 43956.9 = x + 26
# 20 layer: 30611.7 = x + 13
# 10 layer: 17800.8 = x
# 10 layer: 16893.2 no offload
