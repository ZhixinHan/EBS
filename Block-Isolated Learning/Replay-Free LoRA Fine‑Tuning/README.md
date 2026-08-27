# Replay-Free LoRA Fine‑Tuning

## Environment

The requirements are provided in `requirements.xlsx`.

## Benchmark

Following previous studies, we use two widely adopted benchmarks: SuperNI [1] and Long Sequence [2] to evaluate EBS. To ensure a fair comparison, we follow [3] by selecting the same 15 tasks and using the same instance sampling strategy for each benchmark. And for each benchmark, we evaluate on the two task orders specified in [3], and all reported results are averaged over two orders. Furthermore, we also ensure consistency in the evaluation metrics used for every task.

You can train on specific task orders by following [3]. The SuperNI benchmark can be found at `benchmarks/SuperNI`. The Long Sequence benchmark can be found at `benchmarks/Long_Sequence`.

[1] Wang, Y.; Mishra, S.; Alipoormolabashi, P.; Kordi, Y.; Mirzaei, A.; Arunkumar, A.; Ashok, A.; Dhanasekaran, A. S.; Naik, A.; Stap, D.; Pathak, E.; Karamanolakis, G.; Lai, H. G.; Purohit, I.; Mondal, I.; Anderson, J.; Kuznia, K.; Doshi, K.; Patel, M.; Pal, K. K.; Moradshahi, M.; Parmar, M.; Purohit, M.; Varshney, N.; Kaza, P. R.; Verma, P.; Puri, R. S.; Karia, R.; Sampat, S. K.; Doshi, S.; Mishra, S.; Reddy, S.; Patro, S.; Dixit, T.; Shen, X.; Baral, C.; Choi, Y.; Smith, N. A.; Hajishirzi, H.; and Khashabi, D. 2022a. Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks. arXiv:2204.07705.

[2] Razdaibiedina, A.; Mao, Y.; Hou, R.; Khabsa, M.; Lewis, M.; and Almahairi, A. 2023. Progressive Prompts: Continual Learning for Language Models. arXiv:2301.12314.

[3] Zhao, W.; Wang, S.; Hu, Y.; Zhao, Y.; Qin, B.; Zhang, X.; Yang, Q.; Xu, D.; and Che, W. 2024. SAPT: A Shared Attention Framework for Parameter-Efficient Continual Learning of Large Language Models. arXiv:2401.08295.

## Download the Pre-trained Language Model

Download the pre-trained model `google-t5/t5-large` from the Hugging Face website, and place the model files in the folder `models/t5-large`.

## Hyperparameters

The hyperparameter settings for all tasks are provided in the `Hyperparameters` folder.

## Fine-Tune

On line 286 of `train.py`, in the call `wandb.init(entity="", ...)`, the `entity` argument should be set to your **WandB username** or **team name**. Please make sure to properly configure your WandB settings according to your environment; otherwise, the code may fail to run due to incorrect WandB configuration. Then, simply run `SuperNI+t5-large__train.sh`.

## LoRA Blocks

After fine-tuning, a new folder named `logs_and_outputs` will be generated. All LoRA block weights are saved under `logs_and_outputs/SuperNI+t5-large/<timestamp>/`, where `<timestamp>` corresponds to the date and time when you ran the fine-tuning script.

