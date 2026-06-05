import streamlit as st
import os

st.set_page_config(
    page_title="Deteksi Plagiarisme Semantik",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
css_path = os.path.join(os.path.dirname(__file__), 'static', 'styles.css')
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Page Header
st.markdown("""
<div class="page-header">
    <h1>🔍 Deteksi Plagiarisme Semantik</h1>
    <p>Studi Komparatif Multilingual Sentence-BERT dan IndoBERT pada Teks Bahasa Indonesia</p>
</div>
""", unsafe_allow_html=True)

# Model Cards
st.markdown("## Model yang Digunakan")

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
            <p>Baseline klasik yang menghitung frekuensi kata terbobot. Cepat dan ringan, cocok sebagai pembanding awal.</p>
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
            <p>Sentence-BERT multilingual yang menghasilkan embedding semantik. Memahami konteks lintas bahasa.</p>
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
            <p>Model BERT yang dilatih khusus untuk Bahasa Indonesia. Optimal untuk teks berbahasa Indonesia.</p>
            <div class="model-badge model-badge-indobert">Indonesia</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Navigation
st.markdown("## Navigasi")

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    st.markdown("""
    <a href="pages/1_Demo.py" class="nav-card">
        <span class="nav-card-icon">🚀</span>
        <p class="nav-card-title">Demo</p>
        <p class="nav-card-desc">Coba deteksi plagiarisme secara interaktif</p>
    </a>
    """, unsafe_allow_html=True)

with nav_col2:
    st.markdown("""
    <a href="pages/2_Evaluasi.py" class="nav-card">
        <span class="nav-card-icon">📊</span>
        <p class="nav-card-title">Evaluasi</p>
        <p class="nav-card-desc">Lihat perbandingan performa model</p>
    </a>
    """, unsafe_allow_html=True)

with nav_col3:
    st.markdown("""
    <a href="pages/3_Tentang.py" class="nav-card">
        <span class="nav-card-icon">📖</span>
        <p class="nav-card-title">Tentang</p>
        <p class="nav-card-desc">Pelajari lebih lanjut tentang proyek</p>
    </a>
    """, unsafe_allow_html=True)

st.markdown("---")

# Quick Stats
st.markdown("## Ringkasan Hasil")

stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

with stats_col1:
    st.markdown("""
    <div class="metric-card metric-card-winner">
        <div class="metric-card-label">Model Terbaik</div>
        <div class="metric-card-value">SBERT</div>
        <div class="metric-card-sub">F1-Score: 0.716</div>
    </div>
    """, unsafe_allow_html=True)

with stats_col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-card-label">AUC-ROC Tertinggi</div>
        <div class="metric-card-value">0.7635</div>
        <div class="metric-card-sub">SBERT</div>
    </div>
    """, unsafe_allow_html=True)

with stats_col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-card-label">Dataset</div>
        <div class="metric-card-value">1.725</div>
        <div class="metric-card-sub">Pasangan kalimat</div>
    </div>
    """, unsafe_allow_html=True)

with stats_col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-card-label">Model Dibanding</div>
        <div class="metric-card-value">3</div>
        <div class="metric-card-sub">TF-IDF, SBERT, IndoBERT</div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="professional-footer">
    <p><strong>Moh. Khairul Umam</strong> · NIM 202310370311448 · NLP B</p>
    <p><a href="https://nlp.kair0s.my.id" target="_blank">nlp.kair0s.my.id</a></p>
</div>
""", unsafe_allow_html=True)
