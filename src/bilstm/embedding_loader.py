import numpy as np

def load_glove_embeddings(glove_path):
    embedding_index = {}
    with open(glove_path, 'r', errors='ignore', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = ''.join(values[:-300])
            coefs = np.asarray(values[-300:], dtype='float32')
            embedding_index[word] = coefs
    return embedding_index

def create_embedding_matrix(word_index, embedding_index, embedding_dim):
    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
