import os
import sys
import torch
import intel_extension_for_pytorch
# import oneccl_bindings_for_pytorch
import argparse
from datasets import load_dataset
from numa import memory, schedule

import logging as logging_native
from deepspeed.accelerator import get_accelerator

print("0.0")

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    #BitsAndBytesConfig,
    #HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)

print("0.1")
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

parser = argparse.ArgumentParser(description='LLama2 fine-tuning')

parser.add_argument('--eval', default=False, action='store_true')
# parser.add_argument('--local-rank', type=int, default=-1)

def main():
    args = parser.parse_args()
    print("1")
    model_name = "NousResearch/Llama-2-7b-chat-hf"
    dataset_name = "mlabonne/guanaco-llama2-1k"
    new_model = "llama-2-70b-miniguanaco"
    device_map = {"": "cpu"}

    #CCL related
    os.environ['MASTER_ADDR'] = str(os.environ.get('MASTER_ADDR', 'localhost'))
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
    os.environ['LOCAL_RANK'] = str(os.environ.get('PMI_RANK', 0))
    os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))

    local_rank=-1
    world_size = int(os.environ['WORLD_SIZE'])
    if world_size>1:
        print("World size: ", world_size)
        local_rank = int(os.environ['LOCAL_RANK'])
        print("Rank: ", local_rank)
        memory.set_membind_nodes(local_rank)
        schedule.run_on_nodes(local_rank)

    # Debug logging
    formatter = logging_native.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    logger_custom = logging_native.getLogger('log_r{0}'.format(local_rank))
    #logger_custom.setLevel(logging_native.CRITICAL)
    hdlr_custom = logging_native.FileHandler('log_r{0}.log'.format(local_rank), mode='w')
    hdlr_custom.setFormatter(formatter)
    logger_custom.addHandler(hdlr_custom)
    #logging_native.basicConfig(handlers=[hdlr_custom])
    logger_custom.propagate = False

    if not args.eval:
        lora_r = 64
        lora_alpha = 16
        lora_dropout = 0.1

        print("2")
        use_4bit = False
        # bnb_4bit_compute_dtype = "bfloat16"
        # bnb_4bit_quant_type = "nf4"
        use_nested_quant = False

        logging.set_verbosity(logging.WARNING)

        output_dir = "./results"
        num_train_epochs = 1
        fp16 = True
        bf16 = False
        per_device_train_batch_size = 2 
        gradient_accumulation_steps = 1
        gradient_checkpointing = False
        max_grad_norm = 0.3
        learning_rate = 2e-4
        weight_decay = 0.001
        # optim = "paged_adamw_32bit"
        optim = "adamw_torch"
        lr_scheduler_type = "cosine"
        max_steps = 10
        warmup_ratio = 0.03
        group_by_length = True
        save_steps = 0
        logging_steps = 25

        max_seq_length = None
        packing = False
        

        print("3")
        dataset = load_dataset(dataset_name, split="train")
        # compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=use_4bit,
        #     bnb_4bit_quant_type=bnb_4bit_quant_type,
        #     bnb_4bit_compute_dtype=compute_dtype,
        #     bnb_4bit_use_double_quant=use_nested_quant,
        # )
        
        training_arguments = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            optim=optim,
            save_steps=save_steps,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            # fp16=fp16,
            #bf16=bf16,
            # bf16_full_eval=bf16,
            use_ipex=True,
            local_rank=local_rank,
            ddp_find_unused_parameters=False,
            use_cpu=True,
            # no_cuda=True,# Don't use. Sets device=CPU
            # # Check transformer/training_args.py line 1775
            # # Data parallel hard codes to CUDA...
            # # line 1390: XPU should also be allowed with BF16
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            group_by_length=group_by_length,
            lr_scheduler_type=lr_scheduler_type,
            ddp_backend='ccl',
            deepspeed='ds_config.json',
            report_to="tensorboard"
        )

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            model_name
            #quantization_config=bnb_config
            #device_map=device_map
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1

        # Load LLaMA tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

        # Load LoRA configuration
        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # peft_config = None
        # Set training parameters
        

        # Set supervised fine-tuning parameters
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            args=training_arguments,
            packing=packing,
        )
        
        get_accelerator().empty_cache()
        mega_bytes = 1024.0 * 1024.0
        string = 'before train memory (MB)'
        string += ' | cpu allocated: {}'.format(
            get_accelerator().memory_allocated() / mega_bytes)
        print(string)
        # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        trainer.train()
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=-1))
        if local_rank<1:
            trainer.model.save_pretrained(new_model)

    else:
        if local_rank<1:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                    #quantization_config=bnb_config
                # device_map=device_map
            )
            model = PeftModel.from_pretrained(base_model, new_model)
            model = model.merge_and_unload()
            # model = base_model
            # Load LLaMA tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

    logging.set_verbosity(logging.CRITICAL)

    if local_rank<1:
        prompt = "What is a large language model"
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
        result = pipe(f"<s>[INST] {prompt} [/INST]")
        print(result[0]['generated_text'])

if __name__ == "__main__":
    main()
