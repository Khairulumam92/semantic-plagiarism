from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SBERTModel:
    def __init__(self, model_name='paraphrase-multilingual-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, batch_size=32):
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)

    def get_similarity(self, text1, text2):
        emb1 = self.model.encode([text1])
        emb2 = self.model.encode([text2])
        return cosine_similarity(emb1, emb2)[0][0]

    def get_similarities_batch(self, pairs):
        texts1, texts2 = zip(*pairs)
        emb1 = self.encode(texts1)
        emb2 = self.encode(texts2)
        sims = []
        for e1, e2 in zip(emb1, emb2):
            sims.append(cosine_similarity([e1], [e2])[0][0])
        return np.array(sims)
