# export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_PROCESS_LAUNCHER=none

# export FI_PROVIDER=tcp
# export CCL_ATL_TRANSPORT=mpi
export CCL_ATL_TRANSPORT=ofi
# export CCL_ATL_SHM=1
# export CCL_WORKER_COUNT=1
export DS_ACCELERATOR=xpu

# # 60 layer: ram rss:159Gi
export TASK_NAME=mrpc
deepspeed --num_gpus=12 ../../transformers/examples/pytorch/text-classification/run_glue_no_trainer.py \
--model_name_or_path ~/nightly/hf_models/Llama-2-70b-hf \
--task_name $TASK_NAME \
--max_length 128 \
--pad_to_max_length \
--per_device_train_batch_size 1 \
--learning_rate 2e-5 \
--num_train_epochs 3 \
--max_train_steps 10 \
--output_dir log/


# bash run_falcon_40b.sh 2>&1 | tee mem_bs8_layer10_ac_loraqkv_nodsac.log
# bash run_falcon_40b.sh 2>&1 | tee long.log
# bash run_falcon_40b.sh 2>&1 | tee llama2_70b_xpu_seq4k.log

# bnb requires: pip install transformers==4.30