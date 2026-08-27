#!/bin/bash

config=(
    "task002_quoref_answer_generation:rougeL"
    "task073_commonsenseqa_answer_generation:rougeL"
    "task181_outcome_extraction:rougeL"
    "task363_sst2_polarity_classification:acc"
    "task511_reddit_tifu_long_text_summarization:rougeL"
    "task591_sciq_answer_generation:rougeL"
    "task639_multi_woz_user_utterance_generation:rougeL"
    "task748_glucose_reverse_cause_event_detection:rougeL"
    "task875_emotion_classification:acc"
    "task1290_xsum_summarization:rougeL"
    "task1510_evalution_relation_extraction:rougeL"
    "task1572_samsum_summary:rougeL"
    "task1590_diplomacy_text_generation:rougeL"
    "task1687_sentiment140_classification:acc"
    "task1729_personachat_generate_next:rougeL"
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
    --benchmark SuperNI \
    --max_num_instances_of_task 0 \
    --task "${task_metric%%:*}" \
    --input_text all \
    --do_fp16 False
  done
done
