# Studi Komparatif Multilingual Sentence-BERT dan IndoBERT untuk Deteksi Plagiarisme Semantik pada Teks Bahasa Indonesia

Proyek ini membandingkan tiga pendekatan untuk mendeteksi plagiarisme semantik pada teks Bahasa Indonesia:
- **TF-IDF** (Baseline klasik)
- **Multilingual Sentence-BERT** (Embedding semantik multilingual)
- **IndoBERT** (Model Bahasa Indonesia)

## Dataset

| Dataset | Sumber | Ukuran | Link |
|---|---|---|---|
| Quora Paraphrase Indonesia | Kaggle | 150k+ pasang | [Download](https://www.kaggle.com/datasets/louisowen6/quora-paraphrasing-bahasa-indonesia-version) |
| MSRP Indonesia | HuggingFace | 5.8k pasang | [Download](https://huggingface.co/datasets/jakartaresearch/id-paraphrase-detection) |

## Struktur Folder

```
semantic-plagiarism/
├── app.py                      # Entry point Streamlit
├── preprocessing.py            # Fungsi cleaning teks
├── model_tfidf.py              # Baseline: TF-IDF + cosine similarity
├── model_sbert.py              # Model 1: SBERT multilingual
├── model_indobert.py           # Model 2: IndoBERT + mean pooling
├── evaluate.py                 # F1, Precision, Recall, AUC-ROC
├── requirements.txt            # Dependencies
├── README.md                   # Dokumentasi
├── .gitignore
├── pages/
│   ├── 1_Demo.py               # Demo interaktif
│   ├── 2_Evaluasi.py           # Tabel metrik + grafik
│   └── 3_Tentang.py            # Penjelasan metode
├── notebooks/
│   ├── 01_EDA_preprocessing.ipynb
│   ├── 02_model_comparison.ipynb
│   └── 03_evaluation_final.ipynb
├── assets/                     # Hasil evaluasi (ROC, CM, CSV)
├── data/                       # Dataset (tidak di-commit)
└── docker/                     # Deployment configuration
```

## Cara Install

```bash
# 1. Clone repository
git clone https://github.com/username/semantic-plagiarism.git
cd semantic-plagiarism

# 2. Buat virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

## Cara Run

### Lokal
```bash
streamlit run app.py
```

### Google Colab
Buka notebook di folder `notebooks/` dan jalankan secara berurutan:
1. `01_EDA_preprocessing.ipynb` — EDA & preprocessing
2. `02_model_comparison.ipynb` — Running 3 model
3. `03_evaluation_final.ipynb` — Evaluasi & grafik

### Docker
```bash
cd docker
docker compose up -d --build
```

## Evaluasi

Hasil evaluasi disimpan di folder `assets/`:
- `hasil_evaluasi.csv` — Tabel F1, Precision, Recall, AUC per model
- `roc_curve.png` — Grafik ROC curve
- `confusion_matrix_*.png` — Confusion matrix per model

## Referensi

- Reimers & Gurevych (2019). Sentence-BERT
- IndoBERT: indobenchmark/indobert-base-p1
- Microsoft Research Paraphrase Corpus (MSRP)
