from transformers import HfArgumentParser
from dataclasses import dataclass
from typing import Optional
import os
import numpy as np
import tarfile
from sklearn.preprocessing import normalize


@dataclass
class Params:
    max_num_instances_of_task: Optional[int] = None
    model_family: Optional[str] = None
    model_name: Optional[str] = None
    input_text: Optional[str] = None


def main():
    parser = HfArgumentParser(Params)
    params = parser.parse_args_into_dataclasses()[0]

    np.seterr(all='raise')

    inputs_path = os.path.join("vectors", str(params.input_text) + "_vectors", str(params.model_family),
                               str(params.model_name))
    outputs_path = os.path.join("pooler_vectors", str(params.input_text) + "_pooler_vectors", str(params.model_family))
    os.makedirs(outputs_path, exist_ok=True)

    vectors_path = []

    for item in os.listdir(inputs_path):
        inner_inputs_path = os.path.join(inputs_path, item)
        for inner_item in os.listdir(inner_inputs_path):
            file_path = os.path.join(inner_inputs_path, inner_item)
            for file_name in os.listdir(file_path):
                vectors_path.append(os.path.join(inputs_path, item, inner_item, file_name))

    for path in vectors_path:
        if params.max_num_instances_of_task > 0:
            max_num_vectors = np.load(path)[:params.max_num_instances_of_task]
        else:
            max_num_vectors = np.load(path)

        mean_vector = np.mean(max_num_vectors, axis=0)

        mean_vector = normalize(mean_vector.reshape(1, -1),
                           norm='l2',
                           copy=False)[0]

        np.save(os.path.join(outputs_path, "+".join(path.split("/")[3:])), mean_vector)

    with tarfile.open(os.path.join(outputs_path, str(params.model_name) + ".tar"), mode="w") as tf:
        for item in os.listdir(outputs_path):
            if not item.endswith('.tar'):
                tf.add(os.path.join(outputs_path, item), arcname=item)

    for item in os.listdir(outputs_path):
        if not item.endswith('.tar'):
            os.remove(os.path.join(outputs_path, item))


if __name__ == '__main__':
    main()
