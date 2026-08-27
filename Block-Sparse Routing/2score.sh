#!/bin/bash

PARENT_DIR="logs_and_outputs"

for parent in "$PARENT_DIR"/*/; do
  run_name="$(basename "$parent")"
  tmp="${run_name#*+}"
  decoder_name=${tmp%%+*}
  for dir in "$parent"/*/; do
  log_root="$(basename "$dir")"
    if [[ "$log_root" == *+2 ]]; then
      current_time=${log_root%+2}
    else
      continue
    fi
  python score.py \
  --decoder_name "$decoder_name" \
  --encoder_name bge-small-en-v1_5 \
  --current_time "$current_time" \
  --benchmark SuperNI \
  --MultiModel scale_scores \
  --order 2 \
  --metric score \
  --average__a_0_t 50.94 \
  --tasks task748_glucose_reverse_cause_event_detection:rougeL,task073_commonsenseqa_answer_generation:rougeL,task1590_diplomacy_text_generation:rougeL,task639_multi_woz_user_utterance_generation:rougeL,task1572_samsum_summary:rougeL,task1687_sentiment140_classification:acc,task591_sciq_answer_generation:rougeL,task363_sst2_polarity_classification:acc,task1510_evalution_relation_extraction:rougeL,task1729_personachat_generate_next:rougeL,task181_outcome_extraction:rougeL,task511_reddit_tifu_long_text_summarization:rougeL,task002_quoref_answer_generation:rougeL,task1290_xsum_summarization:rougeL,task875_emotion_classification:acc
  done
done