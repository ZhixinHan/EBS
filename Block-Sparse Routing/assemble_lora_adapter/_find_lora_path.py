import os


def _find_lora_path(lora_root_path):
    global lora_path
    for item in os.listdir(lora_root_path):
        if item.startswith('checkpoint-'):
            lora_path = os.path.join(lora_root_path, item)
            break

    return lora_path