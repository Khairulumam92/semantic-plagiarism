import streamlit as st

st.set_page_config(
    page_title="Tentang - Deteksi Plagiarisme",
    page_icon="📖",
    layout="wide"
)

# Load CSS
import os
css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'styles.css')
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Page Header
st.markdown("""
<div class="page-header">
    <h1>📖 Tentang Proyek</h1>
    <p>Informasi lengkap mengenai metode, dataset, dan referensi yang digunakan</p>
</div>
""", unsafe_allow_html=True)

# Background
st.markdown("## Latar Belakang")

st.markdown("""
<div class="info-box info-box-info">
    <span class="info-box-icon">ℹ️</span>
    <span>
        Deteksi plagiarisme semantik bertujuan mendeteksi kesamaan makna antar teks meskipun menggunakan kata-kata yang berbeda. 
        Ini lebih sulit daripada deteksi plagiarisme berbasis string matching karena memerlukan pemahaman kontekstual.
    </span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
**Contoh:**
- **Teks A:** "Presiden mengumumkan kebijakan ekonomi baru"
- **Teks B:** "Kepala negara merilis regulasi fresh terkait perekonomian"

Meskipun kata-katanya berbeda, kedua kalimat memiliki makna yang sama.
""")

st.markdown("---")

# Methods
st.markdown("## Metode yang Digunakan")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="professional-card">
        <div class="professional-card-header">
            <div class="professional-card-icon" style="background: var(--tfidf-bg); color: var(--tfidf-color);">
                ⚙️
            </div>
            <h3 class="professional-card-title">TF-IDF</h3>
        </div>
        <div class="professional-card-body">
            <p><strong>Deskripsi:</strong> Baseline klasik yang menghitung frekuensi kata terbobot berdasarkan Term Frequency-Inverse Document Frequency.</p>
            <p><strong>Keunggulan:</strong> Cepat, ringan, interpretable</p>
            <p><strong>Keterbatasan:</strong> Tidak memahami konteks atau sinonim</p>
            <div class="model-badge model-badge-tfidf">Baseline</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="professional-card">
        <div class="professional-card-header">
            <div class="professional-card-icon" style="background: var(--sbert-bg); color: var(--sbert-color);">
                🌐
            </div>
            <h3 class="professional-card-title">Sentence-BERT</h3>
        </div>
        <div class="professional-card-body">
            <p><strong>Deskripsi:</strong> Model embedding kalimat multilingual yang menghasilkan representasi vektor semantik.</p>
            <p><strong>Keunggulan:</strong> Memahami konteks, mendukung banyak bahasa</p>
            <p><strong>Model:</strong> paraphrase-multilingual-mpnet-base-v2</p>
            <div class="model-badge model-badge-sbert">Terbaik</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="professional-card">
        <div class="professional-card-header">
            <div class="professional-card-icon" style="background: var(--indobert-bg); color: var(--indobert-color);">
                🇮🇩
            </div>
            <h3 class="professional-card-title">IndoBERT</h3>
        </div>
        <div class="professional-card-body">
            <p><strong>Deskripsi:</strong> Model BERT yang dilatih khusus pada korpus Bahasa Indonesia.</p>
            <p><strong>Keunggulan:</strong> Optimal untuk teks Indonesia, memahami struktur bahasa lokal</p>
            <p><strong>Model:</strong> indobenchmark/indobert-base-p1</p>
            <div class="model-badge model-badge-indobert">Indonesia</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Dataset
st.markdown("## Dataset")

st.markdown("""
<table class="styled-table">
    <thead>
        <tr>
            <th>Dataset</th>
            <th>Sumber</th>
            <th>Ukuran</th>
            <th>Keterangan</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Quora Paraphrase Indonesia</td>
            <td>Kaggle</td>
            <td>150k+ pasang</td>
            <td>Diterjemahkan dari Quora English menggunakan Google Translate</td>
        </tr>
        <tr>
            <td><strong>MSRP Indonesia</strong></td>
            <td>HuggingFace</td>
            <td>5.8k pasang</td>
            <td>Microsoft Research Paraphrase Corpus yang diterjemahkan ke Bahasa Indonesia</td>
        </tr>
    </tbody>
</table>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box info-box-info">
    <span class="info-box-icon">ℹ️</span>
    <span>Dataset MSRP Indonesia digunakan sebagai dataset evaluasi utama dalam proyek ini.</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Evaluation Metrics
st.markdown("## Metrik Evaluasi")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown("""
    <div class="professional-card">
        <div class="professional-card-header">
            <div class="professional-card-icon" style="background: var(--info-bg); color: var(--info);">
                📊
            </div>
            <h3 class="professional-card-title">F1-Score</h3>
        </div>
        <div class="professional-card-body">
            <p>Harmonic mean dari precision dan recall. Metrik utama untuk menilai keseimbangan antara akurasi positif dan kemampuan deteksi.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="professional-card">
        <div class="professional-card-header">
            <div class="professional-card-icon" style="background: var(--success-bg); color: var(--success);">
                ✓
            </div>
            <h3 class="professional-card-title">Precision</h3>
        </div>
        <div class="professional-card-body">
            <p>Proporsi prediksi positif yang benar. Mengukur seberapa akurat model saat mendeteksi plagiarisme.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="professional-card">
        <div class="professional-card-header">
            <div class="professional-card-icon" style="background: var(--warning-bg); color: var(--warning);">
                🔍
            </div>
            <h3 class="professional-card-title">Recall</h3>
        </div>
        <div class="professional-card-body">
            <p>Proporsi positif aktual yang terdeteksi. Mengukur seberapa banyak kasus plagiarisme yang berhasil ditemukan.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown("""
    <div class="professional-card">
        <div class="professional-card-header">
            <div class="professional-card-icon" style="background: var(--danger-bg); color: var(--danger);">
                📈
            </div>
            <h3 class="professional-card-title">AUC-ROC</h3>
        </div>
        <div class="professional-card-body">
            <p>Kemampuan model membedakan kelas positif dan negatif. Nilai 1.0 = sempurna, 0.5 = acak.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# References
st.markdown("## Referensi")

st.markdown("""
<div class="professional-card">
    <div class="professional-card-body">
        <ul style="margin: 0; padding-left: 1.25rem;">
            <li>Reimers, N. & Gurevych, I. (2019). <em>Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks</em>. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing.</li>
            <li>IndoBERT: <em>indobenchmark/indobert-base-p1</em> - Model Bahasa Indonesia oleh IndoBERT Team.</li>
            <li>Microsoft Research Paraphrase Corpus (MSRP) - Dataset standar untuk task paraphrase detection.</li>
            <li>Jakarta AI Research - <em>id-paraphrase-detection</em> dataset di HuggingFace.</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="professional-footer">
    <p><strong>Moh. Khairul Umam</strong> · NIM 202310370311448 · NLP B</p>
    <p><a href="https://nlp.kair0s.my.id" target="_blank">nlp.kair0s.my.id</a></p>
</div>
""", unsafe_allow_html=True)
