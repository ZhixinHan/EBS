import os
from peft import PeftModel
from dataclasses import asdict
import pandas as pd
import json
from ._find_lora_path import _find_lora_path


def assemble__scale_scores(task_dict, params, decoder_model, logs_and_outputs_path):
    task_dict_iter = iter(task_dict)
    first_task = next(task_dict_iter)
    lora_root_path = os.path.join("LoRA_model", params.benchmark + "+" + params.decoder_name)
    lora_root_path_of_first_task = os.path.join(lora_root_path, first_task)

    lora_model = PeftModel.from_pretrained(model=decoder_model, model_id=_find_lora_path(lora_root_path_of_first_task),
                                           adapter_name=first_task)

    for dataset in task_dict_iter:
        _ = lora_model.load_adapter(_find_lora_path(os.path.join(lora_root_path, dataset)), adapter_name=dataset)

    lora_model.add_weighted_adapter(list(task_dict)[:2], [1.0, 0.0], "cat", combination_type="cat")

    for name, param in lora_model.named_parameters():
        if 'lora' in name:
            param.requires_grad = False

    trainable_params, _ = lora_model.get_nb_trainable_parameters()
    assert trainable_params == 0, "The current model has trainable parameters!"

    df = pd.DataFrame(asdict(layer) for layer in lora_model.get_layer_status())
    df.to_excel(os.path.join(logs_and_outputs_path, "layer_status.xlsx"), index=False)
    model_status = asdict(lora_model.get_model_status())

    with open(os.path.join(logs_and_outputs_path, "model_status.json"), 'w') as json_file:
        json.dump(model_status, json_file, indent=4)

    return lora_model
