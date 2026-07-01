# Studi Komparatif Multilingual Sentence-BERT dan IndoBERT untuk Deteksi Plagiarisme Semantik pada Teks Bahasa Indonesia

| | |
|---|---|
| **Nama** | Moh. Khairul Umam |
| **NIM** | 202310370311448 |
| **Kelas** | NLP B |

**Dashboard Tugas Akhir:** [nlp.kair0s.my.id](https://nlp.kair0s.my.id)

Proyek ini membandingkan tiga pendekatan untuk mendeteksi plagiarisme semantik pada teks Bahasa Indonesia:
- **TF-IDF** (Baseline klasik)
- **Multilingual Sentence-BERT** (Embedding semantik multilingual)
- **IndoBERT** (Model Bahasa Indonesia)

## Dataset

| Dataset | Sumber | Ukuran | Link |
|---|---|---|---|
| Quora Paraphrase Indonesia | Kaggle | 150k+ pasang | [Download](https://www.kaggle.com/datasets/louisowen6/quora-paraphrasing-bahasa-indonesia-version) |
| MSRP Indonesia | HuggingFace | 5.8k pasang | [Download](https://huggingface.co/datasets/jakartaresearch/id-paraphrase-detection) |

> Dataset MSRP Indonesia sudah tersedia di folder `data/`. Dataset Quora Indonesia bersifat opsional dan dapat ditambahkan untuk eksperimen lebih lanjut.

## Struktur Folder

```
semantic-plagiarism/
├── app.py                      # Entry point Streamlit (halaman beranda)
├── preprocessing.py            # Fungsi cleaning teks (lowercase, hapus URL/mention/tanda baca)
├── model_tfidf.py              # Baseline: TF-IDF + cosine similarity
├── model_sbert.py              # Model 1: Multilingual SBERT + cosine similarity
├── model_indobert.py           # Model 2: IndoBERT + mean pooling + cosine similarity
├── evaluate.py                 # Fungsi evaluasi: F1, Precision, Recall, AUC-ROC, threshold tuning
├── run_pipeline.py             # Script lengkap: download data → training → evaluasi (all-in-one)
├── requirements.txt            # Dependencies Python
├── README.md                   # Dokumentasi proyek
├── .gitignore
├── static/
│   └── styles.css              # Design system untuk Streamlit (dark mode, responsive)
├── pages/
│   ├── 1_Demo.py               # Demo interaktif: input 2 teks → pilih model → hasil
│   ├── 2_Evaluasi.py           # Tabel metrik + grafik ROC curve + confusion matrix
│   └── 3_Tentang.py            # Penjelasan metode, dataset, dan referensi
├── notebooks/
│   └── notebook.ipynb          # Notebook lengkap: EDA, model comparison, evaluasi + visualisasi
├── assets/
│   ├── roc_curve.png               # Grafik ROC curve ketiga model
│   ├── confusion_matrix_tfidf.png  # Confusion matrix TF-IDF
│   ├── confusion_matrix_sbert.png  # Confusion matrix SBERT
│   ├── confusion_matrix_indobert.png # Confusion matrix IndoBERT
│   ├── metrics_comparison.png      # Grafik perbandingan metrik
│   ├── label_distribution.png      # Distribusi label dataset
│   ├── text_length_distribution.png # Distribusi panjang teks
│   ├── word_overlap_distribution.png # Distribusi word overlap
│   └── hasil_evaluasi.csv          # Tabel F1, Precision, Recall, AUC per model
├── data/
│   ├── id_msrp_train.csv           # Dataset train MSRP Indonesia (4.076 baris)
│   ├── id_msrp_val.csv             # Dataset validation MSRP Indonesia (1.725 baris)
│   ├── msrp_train_clean.csv        # Data train setelah preprocessing
│   ├── msrp_val_clean.csv          # Data validation setelah preprocessing
│   └── similarities.pkl            # Hasil similarity score ketiga model
└── docker/
    ├── Dockerfile                  # Build image Streamlit app
    ├── docker-compose.yml          # Orkestrasi container + Nginx
    ├── nginx.conf                  # Konfigurasi Nginx reverse proxy
    └── .env.example                # Template environment variable
```

## Cara Install

```bash
# 1. Clone repository
git clone https://github.com/Khairulumam92/semantic-plagiarism.git
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

### Opsi 1: All-in-One Pipeline (Tercepat)
```bash
python run_pipeline.py
```
Script ini menjalankan seluruh proses: load dataset → preprocessing → running 3 model → evaluasi → generate grafik.

### Opsi 2: Streamlit Dashboard
```bash
streamlit run app.py
```

### Opsi 3: Google Colab
Buka `notebooks/notebook.ipynb` di Google Colab dan jalankan semua sel secara berurutan. Notebook ini mencakup:
1. **EDA & Preprocessing** — Distribusi label, statistik teks, word overlap, cleaning teks
2. **Model Comparison** — Running TF-IDF, SBERT, IndoBERT, hitung similarity
3. **Evaluasi & Visualisasi** — Threshold tuning, F1/AUC, ROC curve, confusion matrix, bar chart

### Opsi 4: Docker
```bash
cd docker
docker compose up -d --build
```

## Hasil Evaluasi

Dataset: MSRP Indonesia validation set (1.725 pasang kalimat)

| Model | F1-Score | Precision | Recall | AUC-ROC | Threshold |
|---|---|---|---|---|---|
| **TF-IDF** | 0.6920 | 0.8414 | 0.5876 | 0.7386 | 0.6622 |
| **SBERT** | **0.7161** | **0.8412** | **0.6234** | **0.7635** | 0.8459 |
| **IndoBERT** | 0.6817 | 0.8271 | 0.5798 | 0.7242 | 0.8723 |

**Kesimpulan:** SBERT menghasilkan performa terbaik dengan F1-Score 0.716 dan AUC-ROC 0.763. Semua model memiliki precision tinggi (>0.82) namun recall masih moderat (~58-62%), menunjukkan model bersifat konservatif (minim false positive).

Hasil evaluasi visual tersedia di folder `assets/`:
- `roc_curve.png` — Grafik ROC curve perbandingan ketiga model
- `confusion_matrix_*.png` — Confusion matrix per model
- `metrics_comparison.png` — Bar chart perbandingan metrik
- `hasil_evaluasi.csv` — Tabel metrik lengkap

## Referensi

- Reimers & Gurevych (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
- IndoBERT: indobenchmark/indobert-base-p1
- Microsoft Research Paraphrase Corpus (MSRP)
- Jakarta AI Research — id-paraphrase-detection
