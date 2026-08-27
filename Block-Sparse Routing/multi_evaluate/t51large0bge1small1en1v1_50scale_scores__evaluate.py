from dataclasses import dataclass
from typing import Optional, Dict
from transformers import PreTrainedTokenizerBase
import numpy as np
import re
import evaluate
import json
import os
from datasets.download import DownloadConfig
from datasets import DownloadMode, Dataset, DatasetDict
from statistics import mean
import pandas as pd


@dataclass
class EvaluatorForCL:
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    save_path: Optional[str] = None
    raw_dataset: Optional[Dataset] = None
    task_dict: Optional[Dict] = None

    def __call__(self, pred):
        global results
        predictions = pred.predictions
        label_ids = pred.label_ids
        predictions = np.where(predictions == -100, self.tokenizer.pad_token_id, predictions)
        predictions_decode = self.tokenizer.batch_decode(predictions, skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=True)
        label_ids = np.where(label_ids == -100, self.tokenizer.pad_token_id, label_ids)
        label_ids_decode = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=True)

        results = self._compute_metrics(predictions_decode, label_ids_decode, self.raw_dataset, self.task_dict,
                                        self.save_path)

        self._summary(os.path.join(self.save_path, "tasks_metric"))
        self._probe_metrics(self.raw_dataset, self.task_dict, self.save_path)

        return results

    def _compute_metrics(self, predictions, labels, raw_dataset, task_dict, save_path):
        assert len(predictions) == len(labels), "len(predictions) != len(labels)"
        predictions_processed = []
        labels_processed = []

        for pred, gold in zip(predictions, labels):
            predictions_processed.append(re.sub(r'\s+', ' ', pred).strip().lower())
            labels_processed.append(re.sub(r'\s+', ' ', gold).strip().lower())

        rouge = evaluate.load(path="rouge", keep_in_memory=True, download_mode=DownloadMode("force_redownload"),
                              download_config=DownloadConfig(local_files_only=True))

        unique_list = list(task_dict)

        dataset_dict = DatasetDict({
            name: raw_dataset.filter(lambda ex, name=name: ex["dataset_name"] == name)
            for name in unique_list
        })

        num_rows_list = []

        for _, ds in dataset_dict.items():
            num_rows_list.append(ds.num_rows)

        predictions_processed_list = []
        labels_processed_list = []
        start = 0

        for n in num_rows_list:
            predictions_processed_list.append(predictions_processed[start:start + n])
            labels_processed_list.append(labels_processed[start:start + n])
            start += n

        aggregate_results = []

        for idx, (task_predictions_processed, task_labels_processed, (task_name, metric_for_best_model)) in enumerate(
                zip(predictions_processed_list, labels_processed_list, task_dict.items()), start=1):
            save_task_path = os.path.join(save_path, "tasks_metric", str(idx).zfill(2) + "-" + task_name)
            os.makedirs(save_task_path, exist_ok=True)

            exact_match = 0

            for pred, gold in zip(task_predictions_processed, task_labels_processed):
                exact_match += int(pred == gold)

            accuracy = round(exact_match / len(task_predictions_processed) * 100, 2)

            rouge_results = rouge.compute(predictions=task_predictions_processed, references=task_labels_processed)

            results = {key: round(float(value) * 100, 2) for key, value in rouge_results.items()}

            results["acc"] = accuracy
            results["score"] = results[metric_for_best_model]
            results["metric"] = metric_for_best_model

            with open(os.path.join(save_task_path, "predictions.json"), "w") as f:
                json.dump(task_predictions_processed, f, indent=4)

            with open(os.path.join(save_task_path, "golds.json"), "w") as f:
                json.dump(task_labels_processed, f, indent=4)

            with open(os.path.join(save_task_path, "metrics.json"), 'w') as json_file:
                json.dump(results, json_file, indent=4)

            results.pop("metric")
            aggregate_results.append(results)

        averages = {k: mean(d[k] for d in aggregate_results) for k in aggregate_results[0]}

        return averages

    def _summary(self, logs_and_outputs_path):
        task_metric = {}

        for item in sorted(os.listdir(logs_and_outputs_path)):

            if item.endswith('.xlsx') or item.endswith('.txt') or item.endswith('.csv'):
                continue

            with open(os.path.join(logs_and_outputs_path, item, "metrics.json"), 'r', encoding='utf-8') as file:
                task_metric[item] = json.load(file)

        df = pd.DataFrame.from_dict(task_metric, orient='index')

        df.to_excel(os.path.join(logs_and_outputs_path, logs_and_outputs_path.split("/")[-1] + ".xlsx"), index=True)

    def _probe_metrics(self, raw_dataset, task_dict, save_path):
        probe_path = os.path.join(save_path, "probe")
        adapter_list = []

        with open(os.path.join(probe_path, "adapters.jsonl"), 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                adapter_list.append(obj)

        scores = np.loadtxt(os.path.join(probe_path, "scores.csv"), delimiter=',')
        unique_list = list(task_dict)

        dataset_dict = DatasetDict({
            name: raw_dataset.filter(lambda ex, name=name: ex["dataset_name"] == name)
            for name in unique_list
        })

        num_rows_list = []

        for _, ds in dataset_dict.items():
            num_rows_list.append(ds.num_rows)

        task_adapter_list = []
        scores_list = []
        start = 0

        for n in num_rows_list:
            task_adapter_list.append(adapter_list[start:start + n])
            scores_list.append(scores[start:start + n])
            start += n

        for idx, (a_task_adapter, a_scores, (task_name, _)) in enumerate(
                zip(task_adapter_list, scores_list, task_dict.items()), start=1):
            save_task_path = os.path.join(probe_path, str(idx).zfill(2) + "-" + task_name)
            os.makedirs(save_task_path, exist_ok=True)

            with open(os.path.join(save_task_path, 'scores.csv'), 'w', encoding='utf-8') as f:
                np.savetxt(f, a_scores, delimiter=',', fmt="%.18e")

            for one_of_a_task_adapter in a_task_adapter:
                with open(os.path.join(save_task_path, 'adapters.jsonl'), 'a', encoding='utf-8') as fw:
                    fw.write(json.dumps(one_of_a_task_adapter, ensure_ascii=False) + '\n')
