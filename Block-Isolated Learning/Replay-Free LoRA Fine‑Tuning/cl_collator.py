from dataclasses import dataclass
from typing import Optional
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorForCL:
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None

    def __call__(self, batch):
        labels = []
        sources = []

        for instance in batch:
            label = instance['output']
            labels.append(label)
            sources.append(self._get_instruction(instance))

        inputs = self.tokenizer(sources, max_length=self.max_source_length, padding="longest", return_tensors="pt",
                                truncation=True)

        with self.tokenizer.as_target_tokenizer():
            outputs = self.tokenizer(labels, max_length=self.max_target_length, padding="longest", return_tensors="pt",
                                     truncation=True)

        outputs_mask = outputs["attention_mask"].bool()
        inputs["labels"] = outputs["input_ids"].masked_fill(~outputs_mask, -100)

        return inputs

    def _get_instruction(self, instance):
        Definition = "Definition: " + instance['definition'] + "\n"
        Input = "Input: " + instance['input'] + "\n"
        Output = "Output: "
        Source = Definition + Input + Output
        return Source
