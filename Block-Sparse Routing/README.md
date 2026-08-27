# Block-Sparse Routing

## Environment

The requirements are provided in `requirements.xlsx`.

## Download the Pre-trained Language Model

Download the pre-trained model `google-t5/t5-large` from the Hugging Face website, and place the model files in the folder `models/decoders/t5/t5-large/`. Download the pre-trained model `BAAI/bge-small-en-v1.5` from the Hugging Face website, and place the model files in the folder `models/encoders/bge/bge-small-en-v1_5/`.

## Paste LoRA block Weights

In `Replay-Free LoRA Fine-Tuning`, we get LoRA block weights in `logs_and_outputs/SuperNI+t5-large/<timestamp>/`. Copy everything in `logs_and_outputs/SuperNI+t5-large/<timestamp>/`to the`LoRA_model/SuperNI+t5-large/`.

## Paste Task Vectors

In `Task Vector Acquisition`, we get task vectors in `pooler_vectors/all_pooler_vectors/bge/bge-small-en-v1_5.tar`. Paste this file into `pooler_vectors/average/all_pooler_vectors/bge/`.

## Inference

1. Run `1parallel.sh` and `2parallel.sh` to test the model in two different task orders.
2. Run `1score.sh` and `2score.sh` to get experiment results in two different task orders.
3. You can find final results in `logs_and_outputs/SuperNI+t5-large+bge-small-en-v1_5+scale_scores/<timestamp>/summary_score.json`.
