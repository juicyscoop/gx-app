import csv
import numpy as np
import pandas as pd

from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine
from scipy.spatial import cKDTree

class Solver:
    def __init__(self, embeddings_csv_filename, limit=1000000):
        self.embeddings_csv_filename = embeddings_csv_filename
        self.limit = limit

        self.phrases = None
        self.word_embeddings = None
        self.phrase_vectors = None
    
    def prepare(self, phrases_csv_filename="../../data/phrases.csv"):
        if self.phrases is None:
            self.phrases = pd.read_csv(phrases_csv_filename, encoding='ISO-8859-1')['Phrases']

        if self.word_embeddings is None:
            self.word_embeddings = self.load_embeddings_from_csv_file(self.embeddings_csv_filename)
            
        if self.phrase_vectors is None:
            self.phrase_vectors =  {phrase: self.phrase_vector(phrase, self.word_embeddings) for phrase in self.phrases}

    def phrase_vector(self, phrase, word_vectors):
        words = phrase.split()
        vector_sum = np.sum([word_vectors[word] for word in words if word in word_vectors], axis=0)
        if np.linalg.norm(vector_sum) > 0:
            return vector_sum / np.linalg.norm(vector_sum)  # Normalize the vector
        else:
            return np.zeros_like(next(iter(word_vectors.values())))

    def write_dict_to_csv(self, res, output_path):
        df = pd.DataFrame.from_dict(res, orient='index')
        df.to_csv(output_path, index=False)
    
    def get_embeddings_from_binary_file(self, embeddings_binary_filename):
        model = KeyedVectors.load_word2vec_format(embeddings_binary_filename, binary=True, limit=self.limit)
        return model
    
    def save_embeddings_to_csv_file(self, filename, model):
        if not filename.endswith('.csv'):
            raise ValueError("The filename must end with '.csv'")

        if model:
            model.save_word2vec_format(filename)
        else:
            raise ValueError("Model not loaded")
        
    def load_embeddings_from_csv_file(self, filepath):
        word_vectors = {}
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=' ')
            next(reader)
            for line in reader:
                word = line[0]
                vector = np.asarray(line[1:], dtype=np.float32)

                word_vectors[word] = vector
        return word_vectors
    
    def calculate_cosine_distances(self):
        keys = list(self.phrase_vectors.keys())
        n = len(keys)
        distances = {}

        for i in range(0, n):
            for j in range(i + 1, n):
                phrase1 = keys[i]
                phrase2 = keys[j]
                vector1 = self.phrase_vectors[phrase1]
                vector2 = self.phrase_vectors[phrase2]
                distance = cosine(vector1, vector2)  # or np.linalg.norm(vector1 - vector2) for Euclidean
                distances[(phrase1, phrase2)] = distance

        return distances
       
    
    def find_closest_match_kdtree(self, input_phrase, kd_tree, word_vectors, phrase_list):
        input_vector = self.phrase_vector(input_phrase, word_vectors)
        distance, index = kd_tree.query(input_vector)
        return phrase_list[index], distance
    

        
    def task1(self, output_path):
        self.prepare()
        assigned = {}
        for phrase in self.phrases:
            for word in phrase.split():
                if word not in assigned:
                    assigned[word] = self.word_embeddings[word]
        self.write_dict_to_csv(assigned, output_path)

    def task2(self, output_path):
        self.prepare()
        distances = self.calculate_cosine_distances(self.phrase_vectors)
        self.write_dict_to_csv(distances, output_path)
    
    def task3(self,  phrase):
        self.prepare()
        vectors_array = np.array(list(self.phrase_vectors.values()))
        kd_tree = cKDTree(vectors_array)
        closest_match, distance = self.find_closest_match_kdtree(phrase, kd_tree, self.word_vectors, self.phrases)
        return closest_match, distance

