import streamlit as st
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import clean_text

st.set_page_config(
    page_title="Demo - Deteksi Plagiarisme",
    page_icon="🚀",
    layout="wide"
)

# Load CSS
css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'styles.css')
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Page Header
st.markdown("""
<div class="page-header">
    <h1>🚀 Demo Deteksi Plagiarisme</h1>
    <p>Masukkan dua teks untuk mengecek kesamaan makna menggunakan model AI</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_sbert_model(use_int8=True):
    from model_sbert import SBERTModel
    return SBERTModel(use_int8=use_int8)

@st.cache_resource
def load_indobert_model(use_int8=True):
    from model_indobert import IndoBERTModel
    return IndoBERTModel(use_int8=use_int8)

csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'hasil_evaluasi.csv')
thresholds = {'TF-IDF': 0.5, 'SBERT': 0.75, 'IndoBERT': 0.75}
if os.path.exists(csv_path):
    df_eval = pd.read_csv(csv_path)
    for _, row in df_eval.iterrows():
        thresholds[row['model']] = row['threshold']

# Input Section
st.markdown("## Input Teks")

col_input, col_config = st.columns([2, 1])

with col_input:
    text1 = st.text_area("Teks 1", placeholder="Masukkan teks pertama di sini...", height=150, key="text1")
    text2 = st.text_area("Teks 2", placeholder="Masukkan teks kedua di sini...", height=150, key="text2")

with col_config:
    st.markdown("### Konfigurasi")
    
    model_choice = st.selectbox(
        "Pilih Model",
        ["TF-IDF", "SBERT", "IndoBERT"],
        help="TF-IDF: cepat & ringan | SBERT: memahami konteks | IndoBERT: optimal untuk Bahasa Indonesia"
    )

    model_info = {
        "TF-IDF": {
            "desc": "Baseline klasik, menghitung frekuensi kata terbobot",
            "icon": "⚙️",
            "badge": "model-badge-tfidf"
        },
        "SBERT": {
            "desc": "Sentence-BERT multilingual, memahami konteks semantik",
            "icon": "🌐",
            "badge": "model-badge-sbert"
        },
        "IndoBERT": {
            "desc": "Model BERT khusus Bahasa Indonesia",
            "icon": "🇮🇩",
            "badge": "model-badge-indobert"
        }
    }

    info = model_info[model_choice]
    
    st.markdown(f"""
    <div class="info-box info-box-info">
        <span class="info-box-icon">{info['icon']}</span>
        <span>{info['desc']}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin: 1rem 0;">
        <span class="model-badge {info['badge']}">{model_choice}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box info-box-warning">
        <span class="info-box-icon">⚠️</span>
        <span>Threshold optimal: <strong>{thresholds[model_choice]:.4f}</strong></span>
    </div>
    """, unsafe_allow_html=True)

    check_btn = st.button("🔍 Cek Similaritas", type="primary", use_container_width=True)

# Results Section
if check_btn:
    if text1.strip() and text2.strip():
        t1 = clean_text(text1)
        t2 = clean_text(text2)

        if model_choice == "TF-IDF":
            with st.spinner("Memproses dengan TF-IDF..."):
                from model_tfidf import TFIDFModel
                model = TFIDFModel()
                model.fit([(t1, t2)])
                score = model.get_similarity(t1, t2)
        elif model_choice == "SBERT":
            with st.spinner("Memuat model SBERT (INT8 quantized, pertama kali mungkin 10-15 detik)..."):
                model = load_sbert_model(use_int8=True)
            with st.spinner("Menghitung similarity dengan SBERT..."):
                score = model.get_similarity(t1, t2)
        else:
            with st.spinner("Memuat model IndoBERT (INT8 quantized, pertama kali mungkin 15-20 detik)..."):
                model = load_indobert_model(use_int8=True)
            with st.spinner("Menghitung similarity dengan IndoBERT..."):
                score = model.get_similarity(t1, t2)

        threshold = thresholds[model_choice]
        is_plagiarized = score >= threshold

        st.markdown("---")
        st.markdown("## Hasil Analisis")

        # Result Card
        if is_plagiarized:
            st.markdown("""
            <div class="result-card result-plagiarism animate-fadeIn">
                <h2>⚠️ TERDETEKSI PLAGIARISME SEMANTIK</h2>
                <p>Kedua teks memiliki kesamaan makna yang signifikan</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card result-clean animate-fadeIn">
                <h2>✅ TIDAK TERDETEKSI PLAGIARISME SEMANTIK</h2>
                <p>Kedua teks memiliki makna yang berbeda</p>
            </div>
            """, unsafe_allow_html=True)

        # Metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.markdown(f"""
            <div class="metric-card animate-fadeIn">
                <div class="metric-card-label">Skor Similaritas</div>
                <div class="metric-card-value">{score:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        with metric_col2:
            st.markdown(f"""
            <div class="metric-card animate-fadeIn">
                <div class="metric-card-label">Threshold</div>
                <div class="metric-card-value">{threshold:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        with metric_col3:
            margin = score - threshold
            margin_class = "success" if margin >= 0 else "danger"
            st.markdown(f"""
            <div class="metric-card animate-fadeIn">
                <div class="metric-card-label">Selisih</div>
                <div class="metric-card-value">{margin:+.4f}</div>
                <div class="metric-card-sub">{'Di atas threshold' if margin >= 0 else 'Di bawah threshold'}</div>
            </div>
            """, unsafe_allow_html=True)

        # Progress Bar
        progress = min(score / threshold, 1.5) if threshold > 0 else 0
        progress_class = "success" if score >= threshold else "warning" if score >= threshold * 0.8 else "danger"
        
        st.markdown(f"""
        <div style="margin: 1.5rem 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="font-weight: 500;">Progress Skor</span>
                <span style="color: var(--text-muted);">{score:.2%} dari {threshold:.2%}</span>
            </div>
            <div class="progress-bar">
                <div class="progress-bar-fill {progress_class}" style="width: {min(progress * 100, 100):.1f}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Detail Info
        st.markdown(f"""
        <div class="info-box info-box-info">
            <span class="info-box-icon">ℹ️</span>
            <span><strong>Model:</strong> {model_choice} · <strong>Threshold:</strong> {threshold:.4f} · <strong>Skor:</strong> {score:.4f}</span>
        </div>
        """, unsafe_allow_html=True)

        # Preprocessed Text
        with st.expander("🔧 Lihat teks setelah preprocessing", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Teks 1 (cleaned):**")
                st.code(t1 if t1 else "(kosong)")
            with col_b:
                st.markdown("**Teks 2 (cleaned):**")
                st.code(t2 if t2 else "(kosong)")
    else:
        st.markdown("""
        <div class="info-box info-box-warning">
            <span class="info-box-icon">⚠️</span>
            <span>Masukkan kedua teks terlebih dahulu.</span>
        </div>
        """, unsafe_allow_html=True)
