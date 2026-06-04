from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class IndoBERTModel:
    def __init__(self, model_name='indobenchmark/indobert-base-p1', use_int8=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        if use_int8:
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, texts, batch_size=32):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt')
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                model_output = self.model(**encoded)
            batch_embeddings = self.mean_pooling(model_output, encoded['attention_mask'])
            embeddings.append(batch_embeddings.cpu().numpy())
        return np.vstack(embeddings)

    def get_similarity(self, text1, text2):
        emb1 = self.encode([text1])
        emb2 = self.encode([text2])
        return cosine_similarity(emb1, emb2)[0][0]

    def get_similarities_batch(self, pairs):
        texts1, texts2 = zip(*pairs)
        emb1 = self.encode(texts1)
        emb2 = self.encode(texts2)
        sims = []
        for e1, e2 in zip(emb1, emb2):
            sims.append(cosine_similarity([e1], [e2])[0][0])
        return np.array(sims)
