import numpy as np
import pickle


embedding = 'total_word_vectors.pickle'

word_embedding = None

with open(embedding, 'rb') as f:
    word_embedding = pickle.load(f)
    id_to_words = word_embedding['id_to_word']
    embedding = word_embedding['embedding']

    words_to_id = {word_embedding[i]: i for i in id_to_words}




