from sklearn.metrics.pairwise import cosine_similarity


def cosine(vectors_matrix):
    cos_sim_matrix = cosine_similarity(vectors_matrix)
    return cos_sim_matrix


def Inner_Product(vectors_matrix):
    scores = vectors_matrix @ vectors_matrix.T
    return scores
