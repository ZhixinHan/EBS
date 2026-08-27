from dataclasses import dataclass, asdict
from typing import Optional
from transformers import HfArgumentParser, set_seed, AutoTokenizer, T5ForConditionalGeneration, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import disable_caching, load_dataset
import os
from datasets.download import DownloadConfig
import time
from peft import LoraConfig, TaskType, get_peft_model
from cl_collator import DataCollatorForCL
import wandb
from datetime import datetime
from cl_evaluate import EvaluatorForCL
import json
import shutil
import pandas as pd


@dataclass
class Params:
    model_name: Optional[str] = None
    train_batch_size: Optional[int] = None
    eval_batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    learning_rate: Optional[float] = None
    num_train_epochs: Optional[int] = None
    do_bf16: Optional[str] = None
    max_source_length: Optional[int] = None
    generation_max_length: Optional[int] = None
    lr_scheduler_type: Optional[str] = None
    warmup_steps: Optional[int] = None
    logging_strategy: Optional[str] = None
    logging_steps: Optional[int] = None
    metric_for_best_model: Optional[str] = None
    evaluation_strategy: Optional[str] = None
    eval_steps: Optional[int] = None
    save_strategy: Optional[str] = None
    save_steps: Optional[int] = None
    save_total_limit: Optional[int] = None
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None
    do_load_best_model_at_end: Optional[str] = None
    seed: Optional[int] = None
    current_time: Optional[str] = None
    benchmark: Optional[str] = None
    max_num_instances_of_train_task: Optional[int] = None
    max_num_instances_of_eval_task: Optional[int] = None
    max_num_instances_of_test_task: Optional[int] = None
    id: Optional[str] = None
    greater_is_better: Optional[str] = None
    weight_decay: Optional[float] = None
    task: Optional[str] = None


def main():
    global base_model, training_params, trainer, folder_path
    parser = HfArgumentParser(Params)
    params = parser.parse_args_into_dataclasses()[0]
    disable_caching()

    params_dict = asdict(params)

    logs_and_outputs_path = os.path.join("logs_and_outputs", str(params.benchmark) + "+" + str(params.model_name),
                                         str(params.current_time), str(params.task))

    os.makedirs(logs_and_outputs_path, exist_ok=True)
    args_file = os.path.join(logs_and_outputs_path, 'train_args.txt')

    with open(args_file, 'w') as f:
        pass

    for key, value in params_dict.items():
        with open(args_file, 'a') as f:
            f.write(f"--{key} {value}\n")

    # Set seed before initializing model.
    set_seed(params.seed)

    download_config_for_load_dataset = DownloadConfig(local_files_only=True)
    timestamp = str(time.time())

    raw_datasets = load_dataset(
        path="cl_dataset.py",
        data_dir=os.path.join("benchmarks", str(params.benchmark), str(params.task)),
        download_config=download_config_for_load_dataset,
        download_mode="force_redownload",
        verification_mode='no_checks',
        keep_in_memory=True,
        save_infos=False,
        trust_remote_code=True,
        max_num_instances_of_train_task=params.max_num_instances_of_train_task,
        max_num_instances_of_eval_task=params.max_num_instances_of_eval_task,
        max_num_instances_of_test_task=params.max_num_instances_of_test_task,
        timestamp=timestamp
    )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join("models", params.model_name),
                                              use_fast=True)

    lora_config = LoraConfig(r=params.lora_r, target_modules=['q', 'v'], lora_alpha=params.lora_alpha,
                             lora_dropout=params.lora_dropout, task_type=TaskType.SEQ_2_SEQ_LM)

    match params.model_name:
        case "t5-large":
            base_model = T5ForConditionalGeneration.from_pretrained(os.path.join("models", params.model_name),
                                                                    local_files_only=True)
        case "t5-3b":
            base_model = T5ForConditionalGeneration.from_pretrained(os.path.join("models", params.model_name),
                                                                    local_files_only=True)
        case "t5-11b":
            base_model = T5ForConditionalGeneration.from_pretrained(os.path.join("models", params.model_name),
                                                                    local_files_only=True)
        case "t5-base":
            base_model = T5ForConditionalGeneration.from_pretrained(os.path.join("models", params.model_name),
                                                                    local_files_only=True)
        case "t5-small":
            base_model = T5ForConditionalGeneration.from_pretrained(os.path.join("models", params.model_name),
                                                                    local_files_only=True)

    lora_model = get_peft_model(base_model, lora_config)

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]

    data_collator = DataCollatorForCL(tokenizer=tokenizer, max_source_length=params.max_source_length,
                                      max_target_length=params.generation_max_length)

    match params.model_name:
        case "t5-large":
            training_params = Seq2SeqTrainingArguments(
                per_device_train_batch_size=params.train_batch_size,
                per_device_eval_batch_size=params.eval_batch_size,
                gradient_accumulation_steps=params.gradient_accumulation_steps,
                learning_rate=params.learning_rate,
                num_train_epochs=params.num_train_epochs,
                bf16=params.do_bf16.lower() == "true",
                generation_max_length=params.generation_max_length,
                lr_scheduler_type=params.lr_scheduler_type,
                warmup_steps=params.warmup_steps,
                logging_strategy=params.logging_strategy,
                logging_steps=params.logging_steps,
                metric_for_best_model=params.metric_for_best_model,
                evaluation_strategy=params.evaluation_strategy,
                eval_steps=params.eval_steps,
                save_strategy=params.save_strategy,
                save_steps=params.save_steps,
                save_total_limit=params.save_total_limit,
                load_best_model_at_end=params.do_load_best_model_at_end.lower() == "true",
                seed=params.seed,
                predict_with_generate=True,
                output_dir=logs_and_outputs_path,
                logging_dir=logs_and_outputs_path,
                do_train=True,
                do_eval=True,
                do_predict=False,
                report_to="wandb",
                greater_is_better=params.greater_is_better.lower() == "true",
                remove_unused_columns=False
            )
        case "t5-3b":
            training_params = Seq2SeqTrainingArguments(
                per_device_train_batch_size=params.train_batch_size,
                per_device_eval_batch_size=params.eval_batch_size,
                gradient_accumulation_steps=params.gradient_accumulation_steps,
                learning_rate=params.learning_rate,
                num_train_epochs=params.num_train_epochs,
                bf16=params.do_bf16.lower() == "true",
                generation_max_length=params.generation_max_length,
                lr_scheduler_type=params.lr_scheduler_type,
                warmup_steps=params.warmup_steps,
                logging_strategy=params.logging_strategy,
                logging_steps=params.logging_steps,
                metric_for_best_model=params.metric_for_best_model,
                evaluation_strategy=params.evaluation_strategy,
                eval_steps=params.eval_steps,
                save_strategy=params.save_strategy,
                save_steps=params.save_steps,
                save_total_limit=params.save_total_limit,
                load_best_model_at_end=params.do_load_best_model_at_end.lower() == "true",
                seed=params.seed,
                predict_with_generate=True,
                output_dir=logs_and_outputs_path,
                logging_dir=logs_and_outputs_path,
                do_train=True,
                do_eval=True,
                do_predict=False,
                report_to="wandb",
                greater_is_better=params.greater_is_better.lower() == "true",
                remove_unused_columns=False
            )
        case "t5-11b":
            training_params = Seq2SeqTrainingArguments(
                per_device_train_batch_size=params.train_batch_size,
                per_device_eval_batch_size=params.eval_batch_size,
                gradient_accumulation_steps=params.gradient_accumulation_steps,
                learning_rate=params.learning_rate,
                num_train_epochs=params.num_train_epochs,
                bf16=params.do_bf16.lower() == "true",
                generation_max_length=params.generation_max_length,
                lr_scheduler_type=params.lr_scheduler_type,
                warmup_steps=params.warmup_steps,
                logging_strategy=params.logging_strategy,
                logging_steps=params.logging_steps,
                metric_for_best_model=params.metric_for_best_model,
                evaluation_strategy=params.evaluation_strategy,
                eval_steps=params.eval_steps,
                save_strategy=params.save_strategy,
                save_steps=params.save_steps,
                save_total_limit=params.save_total_limit,
                load_best_model_at_end=params.do_load_best_model_at_end.lower() == "true",
                seed=params.seed,
                predict_with_generate=True,
                output_dir=logs_and_outputs_path,
                logging_dir=logs_and_outputs_path,
                do_train=True,
                do_eval=True,
                do_predict=False,
                report_to="wandb",
                greater_is_better=params.greater_is_better.lower() == "true",
                remove_unused_columns=False
            )
        case "t5-base":
            training_params = Seq2SeqTrainingArguments(
                per_device_train_batch_size=params.train_batch_size,
                per_device_eval_batch_size=params.eval_batch_size,
                gradient_accumulation_steps=params.gradient_accumulation_steps,
                learning_rate=params.learning_rate,
                num_train_epochs=params.num_train_epochs,
                bf16=params.do_bf16.lower() == "true",
                generation_max_length=params.generation_max_length,
                lr_scheduler_type=params.lr_scheduler_type,
                warmup_steps=params.warmup_steps,
                logging_strategy=params.logging_strategy,
                logging_steps=params.logging_steps,
                metric_for_best_model=params.metric_for_best_model,
                evaluation_strategy=params.evaluation_strategy,
                eval_steps=params.eval_steps,
                save_strategy=params.save_strategy,
                save_steps=params.save_steps,
                save_total_limit=params.save_total_limit,
                load_best_model_at_end=params.do_load_best_model_at_end.lower() == "true",
                seed=params.seed,
                predict_with_generate=True,
                output_dir=logs_and_outputs_path,
                logging_dir=logs_and_outputs_path,
                do_train=True,
                do_eval=True,
                do_predict=False,
                report_to="wandb",
                greater_is_better=params.greater_is_better.lower() == "true",
                remove_unused_columns=False
            )
        case "t5-small":
            training_params = Seq2SeqTrainingArguments(
                per_device_train_batch_size=params.train_batch_size,
                per_device_eval_batch_size=params.eval_batch_size,
                gradient_accumulation_steps=params.gradient_accumulation_steps,
                learning_rate=params.learning_rate,
                num_train_epochs=params.num_train_epochs,
                bf16=params.do_bf16.lower() == "true",
                generation_max_length=params.generation_max_length,
                lr_scheduler_type=params.lr_scheduler_type,
                warmup_steps=params.warmup_steps,
                logging_strategy=params.logging_strategy,
                logging_steps=params.logging_steps,
                metric_for_best_model=params.metric_for_best_model,
                evaluation_strategy=params.evaluation_strategy,
                eval_steps=params.eval_steps,
                save_strategy=params.save_strategy,
                save_steps=params.save_steps,
                save_total_limit=params.save_total_limit,
                load_best_model_at_end=params.do_load_best_model_at_end.lower() == "true",
                seed=params.seed,
                predict_with_generate=True,
                output_dir=logs_and_outputs_path,
                logging_dir=logs_and_outputs_path,
                do_train=True,
                do_eval=True,
                do_predict=False,
                report_to="wandb",
                greater_is_better=params.greater_is_better.lower() == "true",
                remove_unused_columns=False
            )

    wandb.init(entity="",
               project=str(params.benchmark) + "+" + str(params.model_name) + "+" + str(params.current_time),
               group=params.task, mode="offline", dir=logs_and_outputs_path, config=training_params.to_dict(),
               id=params.id, job_type="train", name=datetime.now().strftime("%Y.%m.%d--%H:%M:%S"))

    eval_evaluator = EvaluatorForCL(tokenizer=tokenizer, prefix="eval")

    match params.model_name:
        case "t5-large":
            trainer = Seq2SeqTrainer(
                model=lora_model,
                args=training_params,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=eval_evaluator,
            )
        case "t5-3b":
            trainer = Seq2SeqTrainer(
                model=lora_model,
                args=training_params,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=eval_evaluator,
            )
        case "t5-11b":
            trainer = Seq2SeqTrainer(
                model=lora_model,
                args=training_params,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=eval_evaluator,
            )
        case "t5-base":
            trainer = Seq2SeqTrainer(
                model=lora_model,
                args=training_params,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=eval_evaluator,
            )
        case "t5-small":
            trainer = Seq2SeqTrainer(
                model=lora_model,
                args=training_params,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=eval_evaluator,
            )

    trainer.train()

    with open(os.path.join(logs_and_outputs_path, "trainer_log.json"), 'w') as json_file:
        json.dump(trainer.state.log_history, json_file, indent=4)

    eval_log = [trainer.state.log_history[i] for i in range(1, len(trainer.state.log_history), 2)]
    train_log = [trainer.state.log_history[i] for i in range(0, len(trainer.state.log_history), 2)]

    with open(os.path.join(logs_and_outputs_path, "eval_log.json"), 'w') as json_file:
        json.dump(eval_log, json_file, indent=4)

    with open(os.path.join(logs_and_outputs_path, "train_log.json"), 'w') as json_file:
        json.dump(train_log, json_file, indent=4)

    best_eval_loss = float("inf")
    best_eval_rouge1 = float("-inf")
    best_eval_rouge2 = float("-inf")
    best_eval_rougeL = float("-inf")
    best_eval_rougeLsum = float("-inf")
    best_eval_acc = float("-inf")

    best_eval_loss_epoch = 0
    best_eval_rouge1_epoch = 0
    best_eval_rouge2_epoch = 0
    best_eval_rougeL_epoch = 0
    best_eval_rougeLsum_epoch = 0
    best_eval_acc_epoch = 0

    for count, eval_metric in enumerate(eval_log):
        if best_eval_loss > eval_metric["eval_loss"]:
            best_eval_loss = eval_metric["eval_loss"]
            best_eval_loss_epoch = count + 1
        if best_eval_rouge1 < eval_metric["eval_rouge1"]:
            best_eval_rouge1 = eval_metric["eval_rouge1"]
            best_eval_rouge1_epoch = count + 1
        if best_eval_rouge2 < eval_metric["eval_rouge2"]:
            best_eval_rouge2 = eval_metric["eval_rouge2"]
            best_eval_rouge2_epoch = count + 1
        if best_eval_rougeL < eval_metric["eval_rougeL"]:
            best_eval_rougeL = eval_metric["eval_rougeL"]
            best_eval_rougeL_epoch = count + 1
        if best_eval_rougeLsum < eval_metric["eval_rougeLsum"]:
            best_eval_rougeLsum = eval_metric["eval_rougeLsum"]
            best_eval_rougeLsum_epoch = count + 1
        if best_eval_acc < eval_metric["eval_acc"]:
            best_eval_acc = eval_metric["eval_acc"]
            best_eval_acc_epoch = count + 1

    best_train_loss = float("inf")
    best_train_loss_epoch = 0

    for count, train_metric in enumerate(train_log):
        if best_train_loss > train_metric.get("loss", float("inf")):
            best_train_loss = train_metric["loss"]
            best_train_loss_epoch = count + 1

    wandb.run.summary["best_eval_metrics"] = {
        "loss": (best_eval_loss, best_eval_loss_epoch),
        "rouge1": (best_eval_rouge1, best_eval_rouge1_epoch),
        "rouge2": (best_eval_rouge2, best_eval_rouge2_epoch),
        "rougeL": (best_eval_rougeL, best_eval_rougeL_epoch),
        "rougeLsum": (best_eval_rougeLsum, best_eval_rougeLsum_epoch),
        "acc": (best_eval_acc, best_eval_acc_epoch)
    }

    wandb.run.summary["best_train_loss"] = (best_train_loss, best_train_loss_epoch)

    lora_model.print_trainable_parameters()

    for item in os.listdir(logs_and_outputs_path):
        if item.startswith('checkpoint-'):
            folder_path = os.path.join(logs_and_outputs_path, item)

            for content in os.listdir(folder_path):
                content_path = os.path.join(folder_path, content)
                if os.path.isfile(content_path) or os.path.islink(content_path):
                    if not content_path.endswith('trainer_state.json'):
                        os.unlink(content_path)
                elif os.path.isdir(content_path):
                    shutil.rmtree(content_path)
            break

    lora_model.save_pretrained(folder_path)
    df = pd.DataFrame(asdict(layer) for layer in lora_model.get_layer_status())
    df.to_excel(os.path.join(logs_and_outputs_path, "layer_status.xlsx"), index=False)
    model_status = asdict(lora_model.get_model_status())

    with open(os.path.join(logs_and_outputs_path, "model_status.json"), 'w') as json_file:
        json.dump(model_status, json_file, indent=4)

    wandb.finish()

    wandb_path = os.path.join(logs_and_outputs_path, "wandb")

    if os.path.islink(os.path.join(wandb_path, "latest-run")):
        os.unlink(os.path.join(wandb_path, "latest-run"))
    if os.path.islink(os.path.join(wandb_path, "debug-internal.log")):
        os.unlink(os.path.join(wandb_path, "debug-internal.log"))
    if os.path.islink(os.path.join(wandb_path, "debug.log")):
        os.unlink(os.path.join(wandb_path, "debug.log"))
    for item in os.listdir(wandb_path):
        if item.startswith('offline-run-'):
            try:
                os.unlink(os.path.join(wandb_path, item, "logs", "debug-core.log"))
            except FileNotFoundError:
                continue


if __name__ == '__main__':
    main()
