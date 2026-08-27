#!/bin/bash

config=(
    "agnews:acc"
    "amazon:acc"
    "boolq:acc"
    "cb:acc"
    "copa:acc"
    "dbpedia:acc"
    "imdb:acc"
    "mnli:acc"
    "multirc:acc"
    "qqp:acc"
    "rte:acc"
    "sst2:acc"
    "wic:acc"
    "yahoo:acc"
    "yelp:acc"
)

model_name=(
    "bge-small-en-v1_5"
)

for model_size in "${model_name[@]}"; do
  for task_metric in "${config[@]}"; do
    python generate_input_ids.py \
    --model_family bge \
    --model_name "$model_size" \
    --do_bf16 True \
    --max_source_length 512 \
    --benchmark Long_Sequence \
    --max_num_instances_of_task 0 \
    --task "${task_metric%%:*}" \
    --input_text all \
    --do_fp16 False
  done
done
