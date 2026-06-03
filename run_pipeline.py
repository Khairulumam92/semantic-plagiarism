import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pickle
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch

os.makedirs('data', exist_ok=True)
os.makedirs('assets', exist_ok=True)

# ============================================
# STEP 1: LOAD DATASET MSRP INDONESIA
# ============================================
print("=" * 60)
print("STEP 1: Loading MSRP Indonesia dataset from CSV...")
print("=" * 60)

os.makedirs('data', exist_ok=True)

train_url = "https://media.githubusercontent.com/media/jakartaresearch/hf-datasets/main/msrp/id_train.csv"
val_url = "https://media.githubusercontent.com/media/jakartaresearch/hf-datasets/main/msrp/id_test.csv"

if not os.path.exists('data/id_msrp_train.csv'):
    print("Downloading train set...")
    msrp_train = pd.read_csv(train_url)
    msrp_train.to_csv('data/id_msrp_train.csv', index=False)
else:
    msrp_train = pd.read_csv('data/id_msrp_train.csv')

if not os.path.exists('data/id_msrp_val.csv'):
    print("Downloading validation set...")
    msrp_val = pd.read_csv(val_url)
    msrp_val.to_csv('data/id_msrp_val.csv', index=False)
else:
    msrp_val = pd.read_csv('data/id_msrp_val.csv')

print(f"\nTrain: {len(msrp_train)} rows")
print(f"Validation: {len(msrp_val)} rows")
print(f"Columns: {msrp_train.columns.tolist()}")
print(f"\nLabel distribution (train):")
print(msrp_train['label'].value_counts())

# ============================================
# STEP 2: EDA - LABEL DISTRIBUTION
# ============================================
print("\n" + "=" * 60)
print("STEP 2: EDA - Label Distribution")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
msrp_train['label'].value_counts().plot(kind='bar', ax=axes[0], color=['skyblue', 'salmon'])
axes[0].set_title('MSRP Train - Label Distribution')
axes[0].set_xlabel('Label (0=Bukan, 1=Plagiarisme)')
axes[0].set_ylabel('Count')

msrp_val['label'].value_counts().plot(kind='bar', ax=axes[1], color=['skyblue', 'salmon'])
axes[1].set_title('MSRP Validation - Label Distribution')
axes[1].set_xlabel('Label (0=Bukan, 1=Plagiarisme)')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('assets/label_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: assets/label_distribution.png")

# ============================================
# STEP 3: PREPROCESSING
# ============================================
print("\n" + "=" * 60)
print("STEP 3: Preprocessing")
print("=" * 60)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Cleaning text...")
msrp_train['sentence1_clean'] = msrp_train['sentence1'].apply(clean_text)
msrp_train['sentence2_clean'] = msrp_train['sentence2'].apply(clean_text)
msrp_val['sentence1_clean'] = msrp_val['sentence1'].apply(clean_text)
msrp_val['sentence2_clean'] = msrp_val['sentence2'].apply(clean_text)

msrp_train[['sentence1_clean', 'sentence2_clean', 'label']].to_csv('data/msrp_train_clean.csv', index=False)
msrp_val[['sentence1_clean', 'sentence2_clean', 'label']].to_csv('data/msrp_val_clean.csv', index=False)
print(f"Saved: data/msrp_train_clean.csv ({len(msrp_train)} rows)")
print(f"Saved: data/msrp_val_clean.csv ({len(msrp_val)} rows)")

# Use validation set as test set
df_test = msrp_val.copy()
print(f"\nTest set: {len(df_test)} pairs")
print(f"Label distribution:\n{df_test['label'].value_counts()}")

pairs = list(zip(df_test['sentence1_clean'], df_test['sentence2_clean']))
y_true = df_test['label'].values
texts1, texts2 = zip(*pairs)

# ============================================
# STEP 4: MODEL 1 - TF-IDF
# ============================================
print("\n" + "=" * 60)
print("STEP 4: Running TF-IDF Baseline...")
print("=" * 60)

vectorizer = TfidfVectorizer(max_features=10000)
all_texts = list(texts1) + list(texts2)
vectorizer.fit(all_texts)

vec1 = vectorizer.transform(texts1)
vec2 = vectorizer.transform(texts2)
sims_tfidf = cosine_similarity(vec1, vec2).diagonal()

print(f"TF-IDF done. Mean similarity: {sims_tfidf.mean():.4f}")

# ============================================
# STEP 5: MODEL 2 - SBERT
# ============================================
print("\n" + "=" * 60)
print("STEP 5: Loading & Running SBERT...")
print("=" * 60)

sbert_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
print("Encoding with SBERT...")
emb1_sbert = sbert_model.encode(texts1, batch_size=32, show_progress_bar=True)
emb2_sbert = sbert_model.encode(texts2, batch_size=32, show_progress_bar=True)

sims_sbert = np.array([cosine_similarity([e1], [e2])[0][0] for e1, e2 in zip(emb1_sbert, emb2_sbert)])
print(f"SBERT done. Mean similarity: {sims_sbert.mean():.4f}")

# ============================================
# STEP 6: MODEL 3 - INDOBERT
# ============================================
print("\n" + "=" * 60)
print("STEP 6: Loading & Running IndoBERT...")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

tokenizer = AutoTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
indobert_model = AutoModel.from_pretrained('indobenchmark/indobert-base-p1').to(device)
indobert_model.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_indobert(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt')
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            model_output = indobert_model(**encoded)
        batch_emb = mean_pooling(model_output, encoded['attention_mask'])
        embeddings.append(batch_emb.cpu().numpy())
    return np.vstack(embeddings)

print("Encoding with IndoBERT...")
emb1_indobert = encode_indobert(texts1)
emb2_indobert = encode_indobert(texts2)

sims_indobert = np.array([cosine_similarity([e1], [e2])[0][0] for e1, e2 in zip(emb1_indobert, emb2_indobert)])
print(f"IndoBERT done. Mean similarity: {sims_indobert.mean():.4f}")

# Save similarities
results = {
    'y_true': y_true,
    'sims_tfidf': sims_tfidf,
    'sims_sbert': sims_sbert,
    'sims_indobert': sims_indobert
}
with open('data/similarities.pkl', 'wb') as f:
    pickle.dump(results, f)
print("\nSaved: data/similarities.pkl")

# ============================================
# STEP 7: EVALUATION
# ============================================
print("\n" + "=" * 60)
print("STEP 7: Evaluation & Threshold Tuning")
print("=" * 60)

sims = {
    'TF-IDF': sims_tfidf,
    'SBERT': sims_sbert,
    'IndoBERT': sims_indobert
}

def find_optimal_threshold(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]

thresholds = {}
for name, scores in sims.items():
    thresh = find_optimal_threshold(y_true, scores)
    thresholds[name] = thresh
    print(f"{name} - Optimal threshold: {thresh:.4f}")

metrics_list = []
for name, scores in sims.items():
    thresh = thresholds[name]
    y_pred = (scores >= thresh).astype(int)
    
    metrics = {
        'model': name,
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, scores),
        'threshold': thresh
    }
    metrics_list.append(metrics)
    
    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

df_metrics = pd.DataFrame(metrics_list)
print("\n=== SUMMARY TABLE ===")
print(df_metrics.to_string(index=False))

# ============================================
# STEP 8: GENERATE GRAPHS
# ============================================
print("\n" + "=" * 60)
print("STEP 8: Generating Graphs")
print("=" * 60)

# ROC Curve
plt.figure(figsize=(10, 8))
for name, scores in sims.items():
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Perbandingan Model')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('assets/roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: assets/roc_curve.png")

# Confusion Matrix
for name, scores in sims.items():
    thresh = thresholds[name]
    y_pred = (scores >= thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {name}')
    plt.colorbar()
    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.yticks([0, 1], ['Negative', 'Positive'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='red', fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'assets/confusion_matrix_{name.lower()}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: assets/confusion_matrix_{name.lower()}.png")

# Save evaluation results
df_metrics.to_csv('assets/hasil_evaluasi.csv', index=False)
print("\nSaved: assets/hasil_evaluasi.csv")

print("\n" + "=" * 60)
print("SELESAI! Semua file hasil evaluasi ada di folder assets/")
print("=" * 60)
print("\nFile di assets/:")
for f in os.listdir('assets'):
    print(f"  - assets/{f}")
