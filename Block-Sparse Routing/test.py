from dataclasses import dataclass, asdict
from typing import Optional
from transformers import HfArgumentParser, set_seed, AutoTokenizer, T5ForConditionalGeneration, AutoModel, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import disable_caching, load_dataset
import os
import json
from datasets.download import DownloadConfig
import time
from assemble_lora_adapter import assemble__scale_scores
from multi_collators import SuperNI0t51large0bge1small1en1v1_50scale_scores__collator, \
    Long_Sequence0t51large0bge1small1en1v1_50scale_scores__collator
from multi_evaluate import t51large0bge1small1en1v1_50scale_scores__evaluate
import tarfile
import io
import numpy as np
import torch
from multi_models import t51large0bge1small1en1v1_50scale_scores__model


@dataclass
class Params:
    decoder_name: Optional[str] = None
    encoder_name: Optional[str] = None
    per_device_eval_batch_size: Optional[int] = None
    do_bf16: Optional[str] = None
    do_fp16: Optional[str] = None
    decoder_max_source_length: Optional[int] = None
    generation_max_length: Optional[int] = None
    encoder_max_length: Optional[int] = None
    seed: Optional[int] = None
    current_time: Optional[str] = None
    benchmark: Optional[str] = None
    max_num_instances_of_train_task: Optional[int] = None
    max_num_instances_of_eval_task: Optional[int] = None
    max_num_instances_of_test_task: Optional[int] = None
    tasks: Optional[str] = None
    input_text: Optional[str] = None
    MultiModel: Optional[str] = None
    order: Optional[int] = None
    do_train: Optional[str] = None
    do_eval: Optional[str] = None
    do_predict: Optional[str] = None
    do_probe: Optional[str] = None
    threshold: Optional[float] = None
    num_beams: Optional[int] = None
    temperature: Optional[float] = None
    repetition_penalty: Optional[float] = None
    pooler_vectors__method: Optional[str] = None


def main():
    global decoder_model, raw_datasets, decoder_tokenizer, encoder_tokenizer, assemble_model, encoder_model, model, training_params, data_collator, trainer, test_evaluator
    parser = HfArgumentParser(Params)
    params = parser.parse_args_into_dataclasses()[0]
    disable_caching()

    params_dict = asdict(params)

    assert params.per_device_eval_batch_size == 1, "Batch size cannot be greater than 1!"

    task_list = params.tasks.split(",")
    task_dict = {k: v for k, v in (item.split(':', 1) for item in task_list)}
    task_order_list = list(task_dict)
    sequence_and_task = str(len(task_order_list)).zfill(2) + "-" + task_order_list[-1]

    logs_and_outputs_path = os.path.join("logs_and_outputs",
                                         str(params.benchmark) + "+" + str(params.decoder_name) + "+" + str(
                                             params.encoder_name) + "+" + str(params.MultiModel),
                                         str(params.current_time) + "+" + str(params.order), sequence_and_task)

    os.makedirs(logs_and_outputs_path, exist_ok=True)
    args_file = os.path.join(logs_and_outputs_path, 'test_args.txt')

    with open(args_file, 'w') as f:
        pass

    for key, value in params_dict.items():
        with open(args_file, 'a') as f:
            f.write(f"--{key} {value}\n")

    with open(os.path.join(logs_and_outputs_path, 'tasks.json'), 'w', encoding='utf-8') as f:
        json.dump(task_dict, f, ensure_ascii=False, indent=4)

    # Set seed before initializing the model.
    set_seed(params.seed)

    download_config_for_load_dataset = DownloadConfig(local_files_only=True)
    timestamp = str(time.time())

    match params.benchmark:
        case "SuperNI":
            raw_datasets = load_dataset(
                path=os.path.join("multi_datasets", 'SuperNI_dataset.py'),
                data_dir=os.path.join("benchmarks", str(params.benchmark)),
                download_config=download_config_for_load_dataset,
                download_mode="force_redownload",
                verification_mode='no_checks',
                keep_in_memory=True,
                save_infos=False,
                trust_remote_code=True,
                max_num_instances_of_train_task=params.max_num_instances_of_train_task,
                max_num_instances_of_eval_task=params.max_num_instances_of_eval_task,
                max_num_instances_of_test_task=params.max_num_instances_of_test_task,
                tasks=task_dict,
                timestamp=timestamp
            )
        case "Long_Sequence":
            raw_datasets = load_dataset(
                path=os.path.join("multi_datasets", 'Long_Sequence_dataset.py'),
                data_dir=os.path.join("benchmarks", str(params.benchmark)),
                download_config=download_config_for_load_dataset,
                download_mode="force_redownload",
                verification_mode='no_checks',
                keep_in_memory=True,
                save_infos=False,
                trust_remote_code=True,
                max_num_instances_of_train_task=params.max_num_instances_of_train_task,
                max_num_instances_of_eval_task=params.max_num_instances_of_eval_task,
                max_num_instances_of_test_task=params.max_num_instances_of_test_task,
                tasks=task_dict,
                timestamp=timestamp
            )

    decoders_dir = os.path.join("models", "decoders")
    encoders_dir = os.path.join("models", "encoders")

    decoder_model_family = params.decoder_name.split("-")[0]
    encoder_model_family = params.encoder_name.split("-")[0]

    match params.decoder_name:
        case "t5-large":
            decoder_tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=os.path.join(decoders_dir, decoder_model_family, params.decoder_name),
                use_fast=True, local_files_only=True)
        case "t5-small":
            decoder_tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=os.path.join(decoders_dir, decoder_model_family, params.decoder_name),
                use_fast=True, local_files_only=True)
        case "t5-base":
            decoder_tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=os.path.join(decoders_dir, decoder_model_family, params.decoder_name),
                use_fast=True, local_files_only=True)
        case "t5-3b":
            decoder_tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=os.path.join(decoders_dir, decoder_model_family, params.decoder_name),
                use_fast=True, local_files_only=True)
        case "t5-11b":
            decoder_tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=os.path.join(decoders_dir, decoder_model_family, params.decoder_name),
                use_fast=True, local_files_only=True)

    match params.encoder_name:
        case "bge-small-en-v1_5":
            encoder_tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=os.path.join(encoders_dir, encoder_model_family, params.encoder_name),
                use_fast=True, local_files_only=True)

    match params.decoder_name:
        case "t5-large":
            decoder_model = T5ForConditionalGeneration.from_pretrained(
                os.path.join(decoders_dir, decoder_model_family, params.decoder_name), local_files_only=True)

            decoder_model.generation_config.num_beams = params.num_beams
            decoder_model.generation_config.temperature = params.temperature
            decoder_model.generation_config.repetition_penalty = params.repetition_penalty
        case "t5-small":
            decoder_model = T5ForConditionalGeneration.from_pretrained(
                os.path.join(decoders_dir, decoder_model_family, params.decoder_name), local_files_only=True)

            decoder_model.generation_config.num_beams = params.num_beams
            decoder_model.generation_config.temperature = params.temperature
            decoder_model.generation_config.repetition_penalty = params.repetition_penalty
        case "t5-base":
            decoder_model = T5ForConditionalGeneration.from_pretrained(
                os.path.join(decoders_dir, decoder_model_family, params.decoder_name), local_files_only=True)

            decoder_model.generation_config.num_beams = params.num_beams
            decoder_model.generation_config.temperature = params.temperature
            decoder_model.generation_config.repetition_penalty = params.repetition_penalty
        case "t5-3b":
            decoder_model = T5ForConditionalGeneration.from_pretrained(
                os.path.join(decoders_dir, decoder_model_family, params.decoder_name), local_files_only=True)

            decoder_model.generation_config.num_beams = params.num_beams
            decoder_model.generation_config.temperature = params.temperature
            decoder_model.generation_config.repetition_penalty = params.repetition_penalty
        case "t5-11b":
            decoder_model = T5ForConditionalGeneration.from_pretrained(
                os.path.join(decoders_dir, decoder_model_family, params.decoder_name), local_files_only=True)

            decoder_model.generation_config.num_beams = params.num_beams
            decoder_model.generation_config.temperature = params.temperature
            decoder_model.generation_config.repetition_penalty = params.repetition_penalty

    match params.encoder_name:
        case "bge-small-en-v1_5":
            encoder_model = AutoModel.from_pretrained(
                os.path.join(encoders_dir, encoder_model_family, params.encoder_name), local_files_only=True)

    match params.MultiModel:
        case "scale_scores":
            assemble_model = assemble__scale_scores(task_dict, params, decoder_model, logs_and_outputs_path)

    test_dataset = raw_datasets["test"]

    test_n = len(test_dataset)

    test_dataset = test_dataset.add_column("dataset_type", ["test"] * test_n)

    match params.benchmark + "+" + params.decoder_name + "+" + params.encoder_name + "+" + params.MultiModel:
        case "SuperNI+t5-large+bge-small-en-v1_5+scale_scores":
            data_collator = SuperNI0t51large0bge1small1en1v1_50scale_scores__collator.DataCollatorForCL(
                decoder_tokenizer=decoder_tokenizer, encoder_tokenizer=encoder_tokenizer,
                max_source_length=params.decoder_max_source_length, max_target_length=params.generation_max_length,
                encoder_max_length=params.encoder_max_length, input_text=params.input_text)
        case "Long_Sequence+t5-large+bge-small-en-v1_5+scale_scores":
            data_collator = Long_Sequence0t51large0bge1small1en1v1_50scale_scores__collator.DataCollatorForCL(
                decoder_tokenizer=decoder_tokenizer, encoder_tokenizer=encoder_tokenizer,
                max_source_length=params.decoder_max_source_length, max_target_length=params.generation_max_length,
                encoder_max_length=params.encoder_max_length, input_text=params.input_text)

    match params.decoder_name + "+" + params.encoder_name + "+" + params.MultiModel:
        case "t5-large+bge-small-en-v1_5+scale_scores":
            training_params = Seq2SeqTrainingArguments(
                per_device_eval_batch_size=params.per_device_eval_batch_size,
                bf16=params.do_bf16.lower() == "true",
                fp16=params.do_fp16.lower() == "true",
                generation_max_length=params.generation_max_length,
                seed=params.seed,
                predict_with_generate=True,
                output_dir=logs_and_outputs_path,
                logging_dir=logs_and_outputs_path,
                do_train=params.do_train.lower() == "true",
                do_eval=params.do_eval.lower() == "true",
                do_predict=params.do_predict.lower() == "true",
                report_to="none",
                remove_unused_columns=False
            )
        case "t5-small+bge-small-en-v1_5+scale_scores":
            training_params = Seq2SeqTrainingArguments(
                per_device_eval_batch_size=params.per_device_eval_batch_size,
                bf16=params.do_bf16.lower() == "true",
                fp16=params.do_fp16.lower() == "true",
                generation_max_length=params.generation_max_length,
                seed=params.seed,
                predict_with_generate=True,
                output_dir=logs_and_outputs_path,
                logging_dir=logs_and_outputs_path,
                do_train=params.do_train.lower() == "true",
                do_eval=params.do_eval.lower() == "true",
                do_predict=params.do_predict.lower() == "true",
                report_to="none",
                remove_unused_columns=False
            )
        case "t5-base+bge-small-en-v1_5+scale_scores":
            training_params = Seq2SeqTrainingArguments(
                per_device_eval_batch_size=params.per_device_eval_batch_size,
                bf16=params.do_bf16.lower() == "true",
                fp16=params.do_fp16.lower() == "true",
                generation_max_length=params.generation_max_length,
                seed=params.seed,
                predict_with_generate=True,
                output_dir=logs_and_outputs_path,
                logging_dir=logs_and_outputs_path,
                do_train=params.do_train.lower() == "true",
                do_eval=params.do_eval.lower() == "true",
                do_predict=params.do_predict.lower() == "true",
                report_to="none",
                remove_unused_columns=False
            )
        case "t5-3b+bge-small-en-v1_5+scale_scores":
            training_params = Seq2SeqTrainingArguments(
                per_device_eval_batch_size=params.per_device_eval_batch_size,
                bf16=params.do_bf16.lower() == "true",
                fp16=params.do_fp16.lower() == "true",
                generation_max_length=params.generation_max_length,
                seed=params.seed,
                predict_with_generate=True,
                output_dir=logs_and_outputs_path,
                logging_dir=logs_and_outputs_path,
                do_train=params.do_train.lower() == "true",
                do_eval=params.do_eval.lower() == "true",
                do_predict=params.do_predict.lower() == "true",
                report_to="none",
                remove_unused_columns=False
            )
        case "t5-11b+bge-small-en-v1_5+scale_scores":
            training_params = Seq2SeqTrainingArguments(
                per_device_eval_batch_size=params.per_device_eval_batch_size,
                bf16=params.do_bf16.lower() == "true",
                fp16=params.do_fp16.lower() == "true",
                generation_max_length=params.generation_max_length,
                seed=params.seed,
                predict_with_generate=True,
                output_dir=logs_and_outputs_path,
                logging_dir=logs_and_outputs_path,
                do_train=params.do_train.lower() == "true",
                do_eval=params.do_eval.lower() == "true",
                do_predict=params.do_predict.lower() == "true",
                report_to="none",
                remove_unused_columns=False
            )

    match params.decoder_name + "+" + params.encoder_name + "+" + params.MultiModel:
        case "t5-large+bge-small-en-v1_5+scale_scores":
            test_evaluator = t51large0bge1small1en1v1_50scale_scores__evaluate.EvaluatorForCL(
                tokenizer=decoder_tokenizer, raw_dataset=test_dataset, task_dict=task_dict,
                save_path=logs_and_outputs_path)

    pooler_vectors_path = os.path.join("pooler_vectors", params.pooler_vectors__method,
                                       str(params.input_text) + "_pooler_vectors", encoder_model_family,
                                       params.encoder_name + ".tar")

    pooler_vectors_dict = {}
    members = []

    for task in task_dict:
        members.append(params.encoder_name + "+" + params.benchmark + "+" + task + "+" + "train")

    with tarfile.open(pooler_vectors_path, "r") as tar:
        for member in members:
            file_obj = tar.extractfile(member + ".npy")
            buf = io.BytesIO(file_obj.read())
            arr = np.load(buf, allow_pickle=False)
            pooler_vectors_dict[member] = torch.tensor(arr, dtype=torch.float32)

    pooler_vectors_matrix = torch.stack(list(pooler_vectors_dict.values()), dim=0)

    if params.do_probe.lower() == "true":
        probe = os.path.join(logs_and_outputs_path, "probe")
    else:
        probe = None

    match params.decoder_name + "+" + params.encoder_name + "+" + params.MultiModel:
        case "t5-large+bge-small-en-v1_5+scale_scores":
            model = t51large0bge1small1en1v1_50scale_scores__model(assemble_model=assemble_model,
                                                                   encoder_model=encoder_model,
                                                                   pooler_vectors_matrix=pooler_vectors_matrix,
                                                                   task_order_list=task_order_list,
                                                                   threshold=params.threshold, probe=probe)

    match params.decoder_name + "+" + params.encoder_name + "+" + params.MultiModel:
        case "t5-large+bge-small-en-v1_5+scale_scores":
            trainer = Seq2SeqTrainer(
                model=model,
                args=training_params,
                tokenizer=decoder_tokenizer,
                data_collator=data_collator,
                compute_metrics=test_evaluator
            )
        case "t5-small+bge-small-en-v1_5+scale_scores":
            trainer = Seq2SeqTrainer(
                model=model,
                args=training_params,
                tokenizer=decoder_tokenizer,
                data_collator=data_collator,
                compute_metrics=test_evaluator
            )
        case "t5-base+bge-small-en-v1_5+scale_scores":
            trainer = Seq2SeqTrainer(
                model=model,
                args=training_params,
                tokenizer=decoder_tokenizer,
                data_collator=data_collator,
                compute_metrics=test_evaluator
            )
        case "t5-3b+bge-small-en-v1_5+scale_scores":
            trainer = Seq2SeqTrainer(
                model=model,
                args=training_params,
                tokenizer=decoder_tokenizer,
                data_collator=data_collator,
                compute_metrics=test_evaluator
            )
        case "t5-11b+bge-small-en-v1_5+scale_scores":
            trainer = Seq2SeqTrainer(
                model=model,
                args=training_params,
                tokenizer=decoder_tokenizer,
                data_collator=data_collator,
                compute_metrics=test_evaluator
            )

    output = trainer.predict(test_dataset=test_dataset)
    print(output.metrics)
    metrics = output.metrics

    with open(os.path.join(logs_and_outputs_path, "metrics.json"), 'w') as json_file:
        json.dump(metrics, json_file, indent=4)

    exit_file = os.path.join(logs_and_outputs_path, 'exit_0.txt')

    with open(exit_file, 'w') as f:
        pass


if __name__ == '__main__':
    main()
