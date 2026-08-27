from dataclasses import dataclass, field
from typing import Optional
from transformers import PreTrainedTokenizerBase
import numpy as np
import re
import evaluate
import json
import os
from datasets.download import DownloadConfig
from datasets import DownloadMode


@dataclass
class EvaluatorForCL:
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    prefix: str = field(default=None, metadata={"choices": ["eval", "test"]})
    save_path: Optional[str] = None

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

        if self.prefix == "eval":
            results = self._compute_metrics(predictions_decode, label_ids_decode)
        elif self.prefix == "test":
            results = self._compute_metrics(predictions_decode, label_ids_decode, self.save_path)

        return results

    def _compute_metrics(self, predictions, labels, save_path=None):
        assert len(predictions) == len(labels), "len(predictions) != len(labels)"
        predictions_processed = []
        labels_processed = []

        for pred, gold in zip(predictions, labels):
            predictions_processed.append(re.sub(r'\s+', ' ', pred).strip().lower())
            labels_processed.append(re.sub(r'\s+', ' ', gold).strip().lower())

        exact_match = 0

        for pred, gold in zip(predictions_processed, labels_processed):
            exact_match += int(pred == gold)

        accuracy = round(exact_match / len(predictions_processed) * 100, 2)

        rouge = evaluate.load(path="rouge", keep_in_memory=True, download_mode=DownloadMode("force_redownload"),
                              download_config=DownloadConfig(local_files_only=True))

        rouge_results = rouge.compute(predictions=predictions_processed, references=labels_processed)

        results = {key: round(float(value) * 100, 2) for key, value in rouge_results.items()}

        results["acc"] = accuracy

        if save_path is not None:
            with open(os.path.join(save_path, "predictions.json"), "w") as f:
                json.dump(predictions_processed, f, indent=4)

            with open(os.path.join(save_path, "golds.json"), "w") as f:
                json.dump(labels_processed, f, indent=4)

        return results
