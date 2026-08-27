import numpy as np
from tqdm import tqdm


class FlagEmbeddingTrainer:
    def __init__(self, model, data_collator):
        self.model = model
        self.data_collator = data_collator

    def predict(self, test_dataset):
        todos = [test_dataset[i] for i in range(len(test_dataset))]
        embeddings = []

        for batch in tqdm(todos):
            embeddings.append(self.model.encode(self.data_collator([batch]))[0])

        embeddings_matrix = np.array(embeddings)

        return embeddings_matrix
