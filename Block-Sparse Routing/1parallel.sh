#!/bin/bash

decoder_name_list=(
    "t5-large"
)

for decoder_name in "${decoder_name_list[@]}"; do
  bash 1scale_scores.sh "$decoder_name" &
  sleep 5
done

wait