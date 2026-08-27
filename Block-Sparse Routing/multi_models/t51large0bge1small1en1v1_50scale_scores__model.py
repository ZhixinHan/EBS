from torch import nn
from transformers import GenerationMixin, GenerationConfig
import torch
from transformers.utils import ModelOutput, is_accelerate_available
from typing import Any, Dict, Optional
import os
import numpy as np
import inspect
import torch.nn.functional as F
import json

if is_accelerate_available():
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module


class t51large0bge1small1en1v1_50scale_scores__model(nn.Module, GenerationMixin):
    def __init__(self, assemble_model, encoder_model, pooler_vectors_matrix, task_order_list, threshold, probe=None):
        super(t51large0bge1small1en1v1_50scale_scores__model, self).__init__()
        self.assemble_model = assemble_model
        self.encoder_model = encoder_model
        self.pooler_vectors_matrix = pooler_vectors_matrix.cuda()
        self.task_order_list = task_order_list
        self.probe = probe
        self.threshold = torch.tensor(threshold).cuda()

        os.makedirs(self.probe, exist_ok=True)

        self.assemble_model.base_model.generation_config.use_cache = True

        self.can_generate = self.assemble_model.base_model.can_generate
        self.generation_config = self.assemble_model.base_model.generation_config
        self.config = self.assemble_model.base_model.config
        self.base_model_prefix = self.assemble_model.base_model.base_model_prefix
        self.main_input_name = self.assemble_model.base_model.main_input_name
        self._supports_cache_class = self.assemble_model.base_model._supports_cache_class
        self.device = torch.device("cuda")

    def forward(self, input_ids=None, attention_mask=None, labels=None, cache_position=None, past_key_values=None,
                decoder_input_ids=None, encoder_outputs=None, use_cache=None, return_dict=None, scores=None,
                encoder_input_ids=None, encoder_token_type_ids=None, encoder_attention_mask=None):
        choose_model, scores = self.choose_adapter(input_ids=input_ids, scores=scores,
                                                   encoder_input_ids=encoder_input_ids,
                                                   encoder_token_type_ids=encoder_token_type_ids,
                                                   encoder_attention_mask=encoder_attention_mask)

        outputs = choose_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                               cache_position=cache_position, past_key_values=past_key_values,
                               decoder_input_ids=decoder_input_ids, encoder_outputs=encoder_outputs,
                               use_cache=use_cache, return_dict=return_dict)

        outputs.scores = scores

        return outputs

    def _pooling(self, last_hidden_state: torch.Tensor, pooling_method='cls', attention_mask: torch.Tensor = None):
        if pooling_method == 'cls':
            return last_hidden_state[:, 0]
        elif pooling_method == 'mean':
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d
        return None

    def _update_model_kwargs_for_generation(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
            num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        # update past_key_values keeping its naming used in model code
        cache_name, cache = self._extract_past_from_model_output(outputs)
        model_kwargs[cache_name] = cache
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        if model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
        else:
            past_positions = model_kwargs.pop("cache_position")
            new_positions = torch.arange(
                past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
            ).to(past_positions.device)
            model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
        model_kwargs["scores"] = outputs.scores
        return model_kwargs

    def get_encoder(self, encoder_input_ids, encoder_token_type_ids, encoder_attention_mask):
        choose_model, _ = self.choose_adapter(encoder_input_ids=encoder_input_ids,
                                              encoder_token_type_ids=encoder_token_type_ids,
                                              encoder_attention_mask=encoder_attention_mask)

        return choose_model.get_encoder()

    def _prepare_encoder_decoder_kwargs_for_generation(
            self,
            inputs_tensor: torch.Tensor,
            model_kwargs,
            model_input_name: Optional[str],
            generation_config: GenerationConfig,
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder(encoder_input_ids=model_kwargs['encoder_input_ids'],
                                   encoder_token_type_ids=model_kwargs['encoder_token_type_ids'],
                                   encoder_attention_mask=model_kwargs['encoder_attention_mask'])
        # Compatibility with Accelerate big model inference: we need the encoder to outputs stuff on the same device
        # as the inputs.
        if hasattr(self, "hf_device_map"):
            if hasattr(encoder, "_hf_hook"):
                encoder._hf_hook.io_same_device = True
            else:
                add_hook_to_module(encoder, AlignDevicesHook(io_same_device=True))

        # 2. Prepare encoder args and encoder kwargs from model kwargs and generation config.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }
        encoder_kwargs["output_attentions"] = generation_config.output_attentions
        encoder_kwargs["output_hidden_states"] = generation_config.output_hidden_states

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)  # type: ignore

        return model_kwargs

    def choose_adapter(self, input_ids=None, scores=None, encoder_input_ids=None, encoder_token_type_ids=None,
                       encoder_attention_mask=None):
        if scores is None:
            self.assemble_model.set_adapter(self.task_order_list[0])

            for name, param in self.assemble_model.named_parameters():
                if 'lora' in name:
                    param.requires_grad = False

            self.assemble_model.delete_adapter("cat")

            encoder_last_hidden_state = self.encoder_model(input_ids=encoder_input_ids,
                                                           token_type_ids=encoder_token_type_ids,
                                                           attention_mask=encoder_attention_mask,
                                                           return_dict=True).last_hidden_state

            embeddings = self._pooling(
                encoder_last_hidden_state,
                pooling_method='cls',
                attention_mask=encoder_attention_mask
            )

            # normalize the embedding vectors
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

            scores = self.pooler_vectors_matrix @ embeddings.T

            if self.probe is not None and input_ids is not None:
                with open(os.path.join(self.probe, 'scores.csv'), 'a', encoding='utf-8') as f:
                    np.savetxt(f, scores.T.detach().cpu().numpy(), delimiter=',', fmt="%.18e")

            scores = scores.T[0]
            top2 = scores.topk(2)
            indices = top2.indices
            similarity = self.pooler_vectors_matrix[indices[0]] @ self.pooler_vectors_matrix[indices[1]]

            if similarity > self.threshold:
                temperature = (similarity - self.threshold) / (1 - self.threshold)
                logits = (top2.values - 0.6) / (1 - 0.6)
                weights = F.softmax(logits / temperature, dim=0)
            else:
                weights = torch.zeros_like(top2.values)
                weights[torch.argmax(top2.values)] = 1

            idx_list = indices.tolist()

            sublist = [self.task_order_list[i] for i in idx_list]

            if self.probe is not None and input_ids is not None:
                adapter_dict = {}

                for adapter_name, weight in zip(sublist, weights):
                    adapter_dict[adapter_name] = weight.item()

                with open(os.path.join(self.probe, 'adapters.jsonl'), 'a', encoding='utf-8') as fw:
                    fw.write(json.dumps(adapter_dict, ensure_ascii=False) + '\n')

            self.assemble_model.add_weighted_adapter(sublist, weights, "cat", combination_type="cat")
            self.assemble_model.set_adapter("cat")

            for name, param in self.assemble_model.named_parameters():
                if 'lora' in name:
                    param.requires_grad = False

        return self.assemble_model, scores
