import os
import sys
import logging
import argparse

import torch
import datasets
import transformers
from datasets import load_from_disk
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import AutoModelForCausalLM, set_seed, AutoTokenizer

import idr_torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

def main():

    os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID']

    dist.init_process_group(backend='nccl',
            init_method='env://',
            world_size=idr_torch.size,
            rank=idr_torch.rank)
    print('start')

    parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name",   type=str,   required=False, help="HuggingFace Hub model name", default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--path_dataset", type=str,   required=False, help="Path were the dataset will be loaded", default="./datasets/corpus_name/")
    parser.add_argument("--output_dir",   type=str,   required=False, help="Path were the model will be saved", default="./BioMistral-7B/")
    parser.add_argument("--logging_dir",  type=str,   required=False, help="Path were the model will be saved", default="./BioMistral-7B-logs/")
    
    parser.add_argument("--epochs",        type=int,   required=True, default=30) # 3 and 30
    parser.add_argument("--batch_size",    type=int,   required=True, default=32)
    parser.add_argument("--save_steps",    type=int,   required=True, default=80)
    parser.add_argument("--logging_steps", type=int,   required=True, default=10)
    parser.add_argument("--seed",          type=int,   required=True, default=42)
    parser.add_argument("--learning_rate", type=float, required=True, default=2e-05)
    args = parser.parse_args()

    ####################
    # Training Arguments
    ####################
    training_args = transformers.TrainingArguments(
        bf16=True,
        do_eval=False,
        # evaluation_strategy="epoch",
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
        fsdp="full_shard auto_wrap",
        fsdp_transformer_layer_cls_to_wrap='MistralDecoderLayer',
        local_rank=idr_torch.local_rank,
        weight_decay=0.01,
        max_grad_norm=1.0,
    )
    set_seed(training_args.seed)

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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=2048)
    tokenizer.add_special_tokens({'pad_token': '<pad>'})

    ##############
    # Load dataset
    ##############
    tokenized_datasets = load_from_disk(args.path_dataset)

    ############
    # Load model
    ############
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=None,
        use_flash_attention_2=True,
        torch_dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable()

    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

    #########
    # Trainer
    #########
    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_datasets,
        args=training_args,
        data_collator=data_collator,
    )

    model.config.use_cache = False

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
