#!/bin/bash

current_time=$(date "+%Y-%m-%d_%H-%M-%S")

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

for task_metric in "${config[@]}"; do
  python train.py \
  --model_name t5-large \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --gradient_accumulation_steps 2 \
  --learning_rate 0.0003 \
  --num_train_epochs 10 \
  --do_bf16 True \
  --max_source_length 512 \
  --generation_max_length 50 \
  --lr_scheduler_type constant \
  --warmup_steps 0 \
  --logging_strategy steps \
  --logging_steps 100 \
  --metric_for_best_model "${task_metric#*:}" \
  --evaluation_strategy steps \
  --eval_steps 100 \
  --save_strategy steps \
  --save_steps 100 \
  --save_total_limit 1 \
  --lora_r 8 \
  --lora_alpha 32 \
  --lora_dropout 0.0 \
  --do_load_best_model_at_end True \
  --seed 42 \
  --current_time "$current_time" \
  --benchmark Long_Sequence \
  --max_num_instances_of_train_task 0 \
  --max_num_instances_of_eval_task 0 \
  --max_num_instances_of_test_task 0 \
  --id "$(date +%s%N)" \
  --greater_is_better True \
  --weight_decay 0.0 \
  --task "${task_metric%%:*}"
done
