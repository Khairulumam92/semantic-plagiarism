import streamlit as st

st.set_page_config(
    page_title="Tentang - Deteksi Plagiarisme",
    page_icon="📖",
    layout="wide"
)

st.markdown("""
<style>
    .about-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .about-sub {
        color: #666;
        margin-bottom: 1.5rem;
    }
    .method-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-top: 4px solid #4a90d9;
        height: 100%;
    }
    .method-card h3 {
        margin-top: 0;
        color: #1a1a2e;
    }
    .section-box {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .section-box h3 {
        margin-top: 0;
        color: #1a1a2e;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="about-header">Tentang Proyek</p>', unsafe_allow_html=True)
st.markdown('<p class="about-sub">Informasi lengkap mengenai metode, dataset, dan referensi yang digunakan</p>', unsafe_allow_html=True)

st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.markdown("""
### Latar Belakang

Deteksi plagiarisme semantik bertujuan mendeteksi kesamaan makna antar teks meskipun menggunakan kata-kata yang berbeda. Ini lebih sulit daripada deteksi plagiarisme berbasis string matching karena memerlukan pemahaman kontekstual.

Contoh:
- **Teks A:** "Presiden mengumumkan kebijakan ekonomi baru"
- **Teks B:** "Kepala negara merilis regulasi fresh terkait perekonomian"

Meskipun kata-katanya berbeda, kedua kalimat memiliki makna yang sama.
""")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("### Metode yang Digunakan")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="method-card">
        <h3>TF-IDF</h3>
        <p><strong>Deskripsi:</strong> Baseline klasik yang menghitung frekuensi kata terbobot berdasarkan Term Frequency-Inverse Document Frequency.</p>
        <p><strong>Keunggulan:</strong> Cepat, ringan, interpretable</p>
        <p><strong>Keterbatasan:</strong> Tidak memahami konteks atau sinonim</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="method-card" style="border-top-color: #667eea;">
        <h3>Sentence-BERT</h3>
        <p><strong>Deskripsi:</strong> Model embedding kalimat multilingual yang menghasilkan representasi vektor semantik.</p>
        <p><strong>Keunggulan:</strong> Memahami konteks, mendukung banyak bahasa</p>
        <p><strong>Model:</strong> paraphrase-multilingual-mpnet-base-v2</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="method-card" style="border-top-color: #764ba2;">
        <h3>IndoBERT</h3>
        <p><strong>Deskripsi:</strong> Model BERT yang dilatih khusus pada korpus Bahasa Indonesia.</p>
        <p><strong>Keunggulan:</strong> Optimal untuk teks Indonesia, memahami struktur bahasa lokal</p>
        <p><strong>Model:</strong> indobenchmark/indobert-base-p1</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("### Dataset")

st.markdown("""
| Dataset | Sumber | Ukuran | Keterangan |
|---|---|---|---|
| Quora Paraphrase Indonesia | Kaggle | 150k+ pasang | Diterjemahkan dari Quora English menggunakan Google Translate |
| MSRP Indonesia | HuggingFace | 5.8k pasang | Microsoft Research Paraphrase Corpus yang diterjemahkan ke Bahasa Indonesia |

> Dataset MSRP Indonesia digunakan sebagai dataset evaluasi utama dalam proyek ini.
""")

st.markdown("### Metrik Evaluasi")

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown("""
    <div class="section-box">
        <h3>F1-Score</h3>
        <p>Harmonic mean dari precision dan recall. Metrik utama untuk menilai keseimbangan antara akurasi positif dan kemampuan deteksi.</p>
    </div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown("""
    <div class="section-box">
        <h3>Precision</h3>
        <p>Proporsi prediksi positif yang benar. Mengukur seberapa akurat model saat mendeteksi plagiarisme.</p>
    </div>
    """, unsafe_allow_html=True)
with m3:
    st.markdown("""
    <div class="section-box">
        <h3>Recall</h3>
        <p>Proporsi positif aktual yang terdeteksi. Mengukur seberapa banyak kasus plagiarisme yang berhasil ditemukan.</p>
    </div>
    """, unsafe_allow_html=True)
with m4:
    st.markdown("""
    <div class="section-box">
        <h3>AUC-ROC</h3>
        <p>Kemampuan model membedakan kelas positif dan negatif. Nilai 1.0 = sempurna, 0.5 = acak.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("### Referensi")

st.markdown("""
- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing.
- IndoBERT: *indobenchmark/indobert-base-p1* - Model Bahasa Indonesia oleh IndoBERT Team.
- Microsoft Research Paraphrase Corpus (MSRP) - Dataset standar untuk task paraphrase detection.
- Jakarta AI Research - *id-paraphrase-detection* dataset di HuggingFace.
""")

st.markdown("---")

st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem 0;">
    <p><strong>Moh. Khairul Umam</strong> &middot; NIM 202310370311448 &middot; NLP B</p>
    <p><a href="https://nlp.kair0s.my.id" target="_blank">nlp.kair0s.my.id</a></p>
</div>
""", unsafe_allow_html=True)
