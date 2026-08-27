from dataclasses import dataclass
from typing import Optional
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorForCL:
    decoder_tokenizer: Optional[PreTrainedTokenizerBase] = None
    encoder_tokenizer: Optional[PreTrainedTokenizerBase] = None
    max_source_length: Optional[int] = None
    input_text: Optional[str] = None
    max_target_length: Optional[int] = None
    encoder_max_length: Optional[int] = None

    def __call__(self, batch):
        labels = []
        sources = []

        for instance in batch:
            label = instance['output']
            labels.append(label)
            sources.append(self._get_instruction(instance, self.input_text))

        inputs = self.decoder_tokenizer(sources, max_length=self.max_source_length, padding="max_length",
                                        return_tensors="pt", truncation=True, is_split_into_words=False)

        with self.decoder_tokenizer.as_target_tokenizer():
            outputs = self.decoder_tokenizer(labels, max_length=self.max_target_length, padding="max_length",
                                             return_tensors="pt", truncation=True, is_split_into_words=False)

        encoder_inputs = self.encoder_tokenizer(sources, max_length=self.encoder_max_length, padding="max_length",
                                                return_tensors="pt", truncation=True, is_split_into_words=False)

        outputs_mask = outputs["attention_mask"].bool()
        inputs["labels"] = outputs["input_ids"].masked_fill(~outputs_mask, -100)

        inputs["encoder_input_ids"] = encoder_inputs["input_ids"]
        inputs["encoder_token_type_ids"] = encoder_inputs["token_type_ids"]
        inputs["encoder_attention_mask"] = encoder_inputs["attention_mask"]

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
