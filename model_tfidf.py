from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class TFIDFModel:
    def __init__(self, max_features=10000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.is_fitted = False

    def fit(self, texts):
        all_texts = []
        for t1, t2 in texts:
            all_texts.extend([t1, t2])
        self.vectorizer.fit(all_texts)
        self.is_fitted = True

    def get_similarity(self, text1, text2):
        vec1 = self.vectorizer.transform([text1])
        vec2 = self.vectorizer.transform([text2])
        return cosine_similarity(vec1, vec2)[0][0]

    def get_similarities_batch(self, pairs):
        if not self.is_fitted:
            raise ValueError("Model belum di-fit. Panggil fit() terlebih dahulu.")
        texts1, texts2 = zip(*pairs)
        vec1 = self.vectorizer.transform(texts1)
        vec2 = self.vectorizer.transform(texts2)
        return cosine_similarity(vec1, vec2).diagonal()
