import streamlit as st

st.set_page_config(page_title="Tentang", layout="wide")
st.title("Tentang Proyek")

st.markdown("""
## Latar Belakang
Deteksi plagiarisme semantik bertujuan mendeteksi kesamaan makna antar teks meskipun menggunakan kata-kata yang berbeda. Ini lebih sulit daripada deteksi plagiarisme berbasis string matching karena memerlukan pemahaman kontekstual.

## Metode yang Digunakan

| Metode | Deskripsi | Keunggulan |
|---|---|---|
| **TF-IDF** | Baseline klasik, menghitung frekuensi kata terbobot | Cepat, ringan, interpretable |
| **SBERT** | Sentence-BERT multilingual, menghasilkan embedding semantik | Memahami konteks, multilingual |
| **IndoBERT** | BERT yang dilatih khusus untuk Bahasa Indonesia | Optimal untuk teks Indonesia |

## Dataset

| Dataset | Sumber | Ukuran | Link |
|---|---|---|---|
| Quora Paraphrase Indonesia | Kaggle | 150k+ pasang | [Link](https://www.kaggle.com/datasets/louisowen6/quora-paraphrasing-bahasa-indonesia-version) |
| MSRP Indonesia | HuggingFace | 5.8k pasang | [Link](https://huggingface.co/datasets/jakartaresearch/id-paraphrase-detection) |

## Evaluasi
- **F1-Score**: Harmonic mean dari precision dan recall
- **Precision**: Proporsi prediksi positif yang benar
- **Recall**: Proporsi positif aktual yang terdeteksi
- **AUC-ROC**: Kemampuan model membedakan kelas

## Referensi
- Reimers & Gurevych (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
- IndoBERT: indobenchmark/indobert-base-p1
- Microsoft Research Paraphrase Corpus (MSRP)
""")
