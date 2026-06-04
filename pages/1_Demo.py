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

st.markdown("""
<style>
    .demo-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .demo-sub {
        color: #666;
        margin-bottom: 1.5rem;
    }
    .input-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .result-card {
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .result-plagiarism {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
        color: white;
    }
    .result-clean {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
    }
    .model-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .badge-tfidf { background: #e7f5ff; color: #1971c2; }
    .badge-sbert { background: #f3f0ff; color: #7048e8; }
    .badge-indobert { background: #fff0f6; color: #c2255c; }
    .threshold-info {
        background: #fff9db;
        border-left: 4px solid #fcc419;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-top: 1rem;
        font-size: 0.9rem;
    }
    .stTextArea textarea {
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="demo-header">Demo Deteksi Plagiarisme</p>', unsafe_allow_html=True)
st.markdown('<p class="demo-sub">Masukkan dua teks untuk mengecek kesamaan makna menggunakan model AI</p>', unsafe_allow_html=True)

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

col_input, col_config = st.columns([2, 1])

with col_input:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    text1 = st.text_area("Teks 1", placeholder="Masukkan teks pertama di sini...", height=150)
    text2 = st.text_area("Teks 2", placeholder="Masukkan teks kedua di sini...", height=150)
    st.markdown('</div>', unsafe_allow_html=True)

with col_config:
    st.markdown("### Konfigurasi")
    model_choice = st.selectbox(
        "Pilih Model",
        ["TF-IDF", "SBERT", "IndoBERT"],
        help="TF-IDF: cepat & ringan | SBERT: memahami konteks | IndoBERT: optimal untuk Bahasa Indonesia"
    )

    model_descriptions = {
        "TF-IDF": "Baseline klasik, menghitung frekuensi kata terbobot",
        "SBERT": "Sentence-BERT multilingual, memahami konteks semantik",
        "IndoBERT": "Model BERT khusus Bahasa Indonesia"
    }
    st.info(model_descriptions[model_choice], icon="💡")

    st.markdown(f"""
    <span class="model-badge badge-{model_choice.lower().replace('-', '')}">{model_choice}</span>
    <br><br>
    <small>Threshold optimal: <strong>{thresholds[model_choice]:.4f}</strong></small>
    """, unsafe_allow_html=True)

    check_btn = st.button("Cek Similaritas", type="primary", use_container_width=True)

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

        result_class = "result-plagiarism" if is_plagiarized else "result-clean"
        result_icon = "&#9888;" if is_plagiarized else "&#10004;"
        result_text = "TERDETEKSI PLAGIARISME SEMANTIK" if is_plagiarized else "TIDAK TERDETEKSI PLAGIARISME SEMANTIK"

        st.markdown(f"""
        <div class="result-card {result_class}">
            <h2>{result_icon} {result_text}</h2>
        </div>
        """, unsafe_allow_html=True)

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric(label="Skor Similaritas", value=f"{score:.4f}")
        with metric_col2:
            st.metric(label="Threshold", value=f"{threshold:.4f}")
        with metric_col3:
            margin = score - threshold
            st.metric(label="Selisih", value=f"{margin:+.4f}")

        st.markdown('<div class="threshold-info">', unsafe_allow_html=True)
        st.markdown(f"**Model:** {model_choice} &nbsp;|&nbsp; **Threshold:** {threshold:.4f} &nbsp;|&nbsp; **Skor:** {score:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("Lihat teks setelah preprocessing", icon="⚙️"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Teks 1 (cleaned):**")
                st.code(t1 if t1 else "(kosong)")
            with col_b:
                st.markdown("**Teks 2 (cleaned):**")
                st.code(t2 if t2 else "(kosong)")
    else:
        st.warning("Masukkan kedua teks terlebih dahulu.", icon="⚠️")
