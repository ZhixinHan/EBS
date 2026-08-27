#!/bin/bash

model_name=(
    "bge-small-en-v1_5"
)

for model_size in "${model_name[@]}"; do
    python compute_average.py \
    --model_family bge \
    --model_name "$model_size" \
    --max_num_instances_of_task 0 \
    --input_text all
done
