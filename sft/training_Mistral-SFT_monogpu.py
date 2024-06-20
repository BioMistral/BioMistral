import os
import sys
import logging
import argparse

import torch
import transformers
import datasets
from datasets import load_from_disk
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import AutoModelForCausalLM, set_seed, AutoTokenizer, BitsAndBytesConfig

from peft import LoraConfig, PeftConfig

from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    SFTConfig,
    apply_chat_template,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
)
from trl import SFTTrainer

import torch.distributed as dist
import idr_torch

logger = logging.getLogger(__name__)

def main():

    parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name",   type=str,   required=False, help="HuggingFace Hub model name", default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--path_dataset", type=str,   required=False, help="Path where the dataset will be loaded", default="./datasets/corpus_name/")
    parser.add_argument("--output_dir",   type=str,   required=False, help="Path where the model will be saved", default="./BioMistral-BioInstructQA-EN-models/")
    parser.add_argument("--logging_dir",  type=str,   required=False, help="Path where the model will be saved", default="./BioMistral-BioInstructQA-EN-models-logs/")
    
    parser.add_argument("--epochs",        type=int,   required=True, default=3)
    parser.add_argument("--batch_size",    type=int,   required=True, default=4)
    parser.add_argument("--save_steps",    type=int,   required=True, default=100)
    parser.add_argument("--logging_steps", type=int,   required=True, default=10)
    parser.add_argument("--seed",          type=int,   required=True, default=42)
    parser.add_argument("--learning_rate", type=float, required=True, default=2e-05)

    parser.add_argument("--quantization_config",  type=str,   required=False, help="Path where the model will be saved", default="load_in_8bit")

    args = parser.parse_args()

    ####################
    # Training Arguments
    ####################
    training_args = transformers.TrainingArguments(
        bf16=True,
        do_eval=True,
        evaluation_strategy="epoch",
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        lr_scheduler_type="cosine",
        num_train_epochs=args.epochs,
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=int(args.batch_size*2),
        push_to_hub=False,
        remove_unused_columns=True,
        report_to="tensorboard",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=15,
        seed=args.seed,
        tf32=True,
        logging_dir=args.logging_dir,
        logging_first_step=True,
        optim="adamw_torch",
    )

    set_seed(args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.bf16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")
    
    ###########
    # TOKENIZER
    ###########
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=2048, padding=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    ##############
    # Load dataset
    ##############
    tokenized_datasets = load_from_disk(args.path_dataset)
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    ############
    # Load model
    ############
    logger.info("*** Load pretrained model ***")

    if args.quantization_config=='load_in_4bit':
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # For consistency with model weights, we use the same value as `torch_dtype` which is float16 for PEFT models
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
        )
    elif args.quantization_config=='load_in_8bit':
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    model_kwargs = dict(
        trust_remote_code=True,
        use_flash_attention_2=True,
        torch_dtype=torch.float16,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        modules_to_save=None,
    )

    #########
    # Trainer
    #########
    trainer = SFTTrainer(
        model=args.model_name,
        model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="prompt",
        max_seq_length=2048,
        tokenizer=tokenizer,
        packing=True,
        peft_config=peft_config,
    )

    #############
    # Let's go 🚀
    #############
    train_result = trainer.train()

    #######################
    # Save metrics / models
    #######################
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_model(training_args.output_dir)

    kwargs = {
        "finetuned_from": args.model_name,
        "dataset": args.output_dir,
        "dataset_tags": "Biomedical",
        "tags": ["Mistral","Biomedical","LLM"],
    }
    trainer.create_model_card(**kwargs)
    trainer.model.config.use_cache = True
    trainer.model.config.save_pretrained(training_args.output_dir)

def _mp_fn(index):
    main()

if __name__ == "__main__":
    main()
