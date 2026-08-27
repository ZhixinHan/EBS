# EBS: Efficient Block-Sparse Framework for Continual Learning of Large Language Models

By continuously acquiring new knowledge, continual learning enables large language models (LLMs) to remain adaptive, accurate, and relevant as the world evolves. In continual learning, tasks arrive one after another and the model learns them sequentially. As more tasks are introduced, existing methods for learning a new task impose increasing computational demands. However, we find that the computational overhead, which gradually increases as the number of tasks grows, can be mitigated. To limit the continual increase in computational costs, we introduce a simple efficient block-sparse (EBS) framework for continual learning of LLMs. In EBS, we employ a dedicated, task-specific low-rank adaptation (LoRA) block for each task. When training on a new task, only one LoRA block is involved in the computation, while in inference, each input is routed through at most two LoRA blocks. This design ensures that the computational cost remains independent of the number of tasks, allowing EBS to maintain high efficiency even when the sequence of tasks becomes very large. We offer a simple implementation that works across different model sizes and transformer architectures (decoder‑only and encoder‑decoder) while consistently delivering both high efficiency and strong task performance. Experimental results on two widely-used continual learning benchmarks demonstrate that EBS achieves state-of-the-art performance on the T5 and LLaMA 2 models.

## Code Structure and Organization

The EBS codebase consists of three modules: ​**Replay-Free LoRA Fine‑Tuning**​, ​**Task Vector Acquisition**​, and ​**Block-Sparse Routing**​. While these components are closely connected within the overall EBS framework, they are implemented in a fully decoupled manner, resulting in a cleaner code structure and clearer module boundaries. Each module can function independently, yet together they are logically equivalent to the continual learning approach proposed in our paper. For reproducibility, we recommend implementing and executing these modules in the following order:

1. Replay-Free LoRA Fine‑Tuning
2. Task Vector Acquisition
3. Block-Sparse Routing

