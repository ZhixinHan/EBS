from dataclasses import dataclass
from typing import Optional
from transformers import HfArgumentParser, AutoTokenizer, AutoModel, TrainingArguments, Trainer
import os
from datasets import disable_caching, load_dataset
from datasets.download import DownloadConfig
import time
import cl_collators
import numpy as np
from FlagEmbedding import FlagModel
from Trainers.FlagEmbeddingTrainer import FlagEmbeddingTrainer

os.environ["HF_HUB_OFFLINE"] = "1"


@dataclass
class Params:
    benchmark: Optional[str] = None
    model_name: Optional[str] = None
    model_family: Optional[str] = None
    do_bf16: Optional[str] = None
    max_source_length: Optional[int] = None
    task: Optional[str] = None
    max_num_instances_of_task: Optional[int] = None
    input_text: Optional[str] = None
    do_fp16: Optional[str] = None


def main():
    global raw_datasets, tokenizer, model, training_params, data_collator, trainer, s
    parser = HfArgumentParser(Params)
    params = parser.parse_args_into_dataclasses()[0]
    disable_caching()

    outputs_path = os.path.join("vectors", str(params.input_text) + "_vectors", str(params.model_family),
                                str(params.model_name), str(params.benchmark), str(params.task))
    os.makedirs(outputs_path, exist_ok=True)

    download_config_for_load_dataset = DownloadConfig(local_files_only=True)
    timestamp = str(time.time())

    match params.benchmark:
        case "SuperNI":
            raw_datasets = load_dataset(
                path=os.path.join("cl_datasets", 'SuperNI_dataset.py'),
                data_dir=os.path.join("benchmarks", str(params.benchmark), str(params.task)),
                download_config=download_config_for_load_dataset,
                download_mode="force_redownload",
                verification_mode='no_checks',
                keep_in_memory=True,
                save_infos=False,
                trust_remote_code=True,
                max_num_instances_of_train_task=params.max_num_instances_of_task,
                max_num_instances_of_eval_task=params.max_num_instances_of_task,
                max_num_instances_of_test_task=params.max_num_instances_of_task,
                timestamp=timestamp
            )
        case "Long_Sequence":
            raw_datasets = load_dataset(
                path=os.path.join("cl_datasets", 'Long_Sequence_dataset.py'),
                data_dir=os.path.join("benchmarks", str(params.benchmark), str(params.task)),
                download_config=download_config_for_load_dataset,
                download_mode="force_redownload",
                verification_mode='no_checks',
                keep_in_memory=True,
                save_infos=False,
                trust_remote_code=True,
                max_num_instances_of_train_task=params.max_num_instances_of_task,
                max_num_instances_of_eval_task=params.max_num_instances_of_task,
                max_num_instances_of_test_task=params.max_num_instances_of_task,
                timestamp=timestamp
            )

    model_path = os.path.join("models", params.model_family, params.model_name)

    match params.model_family:
        case "deberta":
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, use_fast=True,
                                                      add_prefix_space=True)
        case "bge":
            tokenizer = None

    match params.model_family:
        case "deberta":
            model = AutoModel.from_pretrained(model_path, local_files_only=True)
        case "bge":
            model = FlagModel(model_path, use_fp16=params.do_fp16.lower() == "true")

    match params.benchmark:
        case "SuperNI":
            data_collator = cl_collators.SuperNI_collator.DataCollatorForCL(tokenizer=tokenizer,
                                                                            max_source_length=params.max_source_length,
                                                                            input_text=params.input_text)
        case "Long_Sequence":
            data_collator = cl_collators.Long_Sequence_collator.DataCollatorForCL(tokenizer=tokenizer,
                                                                                  max_source_length=params.max_source_length,
                                                                                  input_text=params.input_text)

    match params.model_family:
        case "deberta":
            training_params = TrainingArguments(
                per_device_eval_batch_size=1,
                bf16=params.do_bf16.lower() == "true",
                output_dir=outputs_path,
                remove_unused_columns=False,
                do_train=False,
                do_eval=False,
                do_predict=True,
                report_to="none"
            )

    match params.model_family:
        case "deberta":
            trainer = Trainer(
                model=model,
                args=training_params,
                tokenizer=tokenizer,
                data_collator=data_collator
            )
        case "bge":
            trainer=FlagEmbeddingTrainer(
                model=model,
                data_collator=data_collator
            )


    for dataset_type in ["train", "validation", "test"]:
        output = trainer.predict(test_dataset=raw_datasets[dataset_type])

        match params.model_family:
            case "deberta":
                s = output.predictions[:, 0, :]
            case "bge":
                s = output

        if dataset_type == "validation":
            dataset_type = "dev"

        np.save(os.path.join(outputs_path, dataset_type + ".npy"), s)


if __name__ == '__main__':
    main()
