from datasets import BuilderConfig, GeneratorBasedBuilder, DatasetInfo, Features, SplitGenerator, Split, Value
import os
import json
import re
from hashlib import md5


class CLConfig(BuilderConfig):
    def __init__(self, max_num_instances_of_train_task, max_num_instances_of_eval_task, max_num_instances_of_test_task,
                 tasks, timestamp, **kwargs):
        super(CLConfig, self).__init__(**kwargs)
        self.max_num_instances_of_train_task = max_num_instances_of_train_task
        self.max_num_instances_of_eval_task = max_num_instances_of_eval_task
        self.max_num_instances_of_test_task = max_num_instances_of_test_task
        self.timestamp = timestamp
        self.tasks = tasks


class CLBuilder(GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = CLConfig

    def _info(self):
        return DatasetInfo(
            features=Features(
                {
                    "dataset_name": Value("string"),
                    "input": Value("string"),
                    "output": Value("string"),
                    "definition": Value("string"),
                    "timestamp": Value("string")
                }
            )
        )

    def _split_generators(self, dl_manager):
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "data_dir": self.config.data_dir,
                    "max_num_instances_of_task": self.config.max_num_instances_of_train_task,
                    "dataset_type": "train",
                    "timestamp": self.config.timestamp,
                    "tasks": self.config.tasks
                }),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={
                    "data_dir": self.config.data_dir,
                    "max_num_instances_of_task": self.config.max_num_instances_of_eval_task,
                    "dataset_type": "dev",
                    "timestamp": self.config.timestamp,
                    "tasks": self.config.tasks
                }),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={
                    "data_dir": self.config.data_dir,
                    "max_num_instances_of_task": self.config.max_num_instances_of_test_task,
                    "dataset_type": "test",
                    "timestamp": self.config.timestamp,
                    "tasks": self.config.tasks
                }),
        ]

    def _generate_examples(self, data_dir, max_num_instances_of_task, dataset_type, timestamp, tasks):
        for dataset in tasks:
            data_file = os.path.join(data_dir, dataset, dataset_type + '.json')
            yield from self._load_benchmark_dataset(dataset_path=data_file, max_num_instances=max_num_instances_of_task,
                                                    dataset_name=dataset, timestamp=timestamp)

    def _load_benchmark_dataset(self, dataset_path, max_num_instances, dataset_name, timestamp):
        data = self._load_dataset(dataset_path)
        definition = data["Definition"]

        if max_num_instances > 0:
            max_num_instances = data['Instances'][:max_num_instances]
        else:
            max_num_instances = data['Instances']

        for instance in max_num_instances:
            example = {
                "dataset_name": dataset_name,
                "input": re.sub(' {2,}', ' ', instance['input']).strip(),
                "output": re.sub(' {2,}', ' ', instance["output"]).strip(),
                "definition": definition.strip(),
                "timestamp": timestamp
            }

            yield timestamp + "###" + md5(example["input"].encode("utf-8")).hexdigest(), example

    def _load_dataset(self, dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        return data
