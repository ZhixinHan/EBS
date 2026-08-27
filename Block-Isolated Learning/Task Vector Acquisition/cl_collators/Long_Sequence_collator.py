from dataclasses import dataclass
from typing import Optional
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorForCL:
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    max_source_length: Optional[int] = None
    input_text: Optional[str] = None

    def __call__(self, batch):
        sources = []

        for instance in batch:
            sources.append(self._get_instruction(instance, self.input_text))

        if self.tokenizer is not None:
            inputs = self.tokenizer(sources, max_length=self.max_source_length, padding="max_length",
                                    return_tensors="pt", truncation=True, is_split_into_words=False)
        else:
            inputs = sources

        return inputs

    def _get_instruction(self, instance, input_text):
        global Source
        match input_text:
            case "all":
                Source = self._get_all(instance)
            case "only_input":
                Source = self._get_only_input(instance)
            case "remove_Definition":
                Source = self._get_remove_Definition(instance)
            case "remove_template":
                Source = self._get_remove_template(instance)

        return Source

    def _get_all(self, instance):
        Definition = "Definition: " + instance['definition'] + "\n"
        Input = "Input: " + instance['input'] + "\n"
        Output = "Output: "
        Source = Definition + Input + Output
        return Source

    def _get_only_input(self, instance):
        return instance["input"]

    def _get_remove_Definition(self, instance):
        Definition = instance['definition'] + "\n"
        Input = "Input: " + instance['input'] + "\n"
        Output = "Output: "
        Source = Definition + Input + Output
        return Source

    def _get_remove_template(self, instance):
        Definition = instance['definition']
        Input = instance['input']
        Source = Definition + Input
        return Source
