import streamlit as st

st.set_page_config(
    page_title="Deteksi Plagiarisme Semantik",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 4px solid #4a90d9;
        margin-bottom: 1rem;
    }
    .card h3 {
        margin-top: 0;
        color: #1a1a2e;
    }
    .nav-link {
        display: inline-block;
        padding: 0.75rem 1.5rem;
        margin: 0.25rem;
        background: #4a90d9;
        color: white;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 500;
        transition: background 0.2s;
    }
    .nav-link:hover {
        background: #357abd;
        color: white;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
        color: #888;
        font-size: 0.85rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Deteksi Plagiarisme Semantik</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Studi Komparatif Multilingual Sentence-BERT dan IndoBERT pada Teks Bahasa Indonesia</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <h3>&#9881; TF-IDF</h3>
        <p>Baseline klasik yang menghitung frekuensi kata terbobot. Cepat dan ringan, cocok sebagai pembanding awal.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card" style="border-left-color: #667eea;">
        <h3>&#127760; SBERT</h3>
        <p>Sentence-BERT multilingual yang menghasilkan embedding semantik. Memahami konteks lintas bahasa.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card" style="border-left-color: #764ba2;">
        <h3>&#127470;&#127475; IndoBERT</h3>
        <p>Model BERT yang dilatih khusus untuk Bahasa Indonesia. Optimal untuk teks berbahasa Indonesia.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("### Navigasi")

nav_col1, nav_col2, nav_col3 = st.columns(3)
with nav_col1:
    st.page_link("pages/1_Demo.py", label="Demo", icon=":rocket:")
with nav_col2:
    st.page_link("pages/2_Evaluasi.py", label="Evaluasi", icon=":chart_with_upwards_trend:")
with nav_col3:
    st.page_link("pages/3_Tentang.py", label="Tentang", icon=":book:")

st.markdown("---")

st.markdown("### Hasil Evaluasi Singkat")

eval_col1, eval_col2, eval_col3, eval_col4 = st.columns(4)
with eval_col1:
    st.metric(label="Model Terbaik", value="SBERT", delta="F1: 0.716")
with eval_col2:
    st.metric(label="AUC-ROC Tertinggi", value="0.7635", delta="SBERT")
with eval_col3:
    st.metric(label="Dataset", value="1.725", delta="pasangan kalimat")
with eval_col4:
    st.metric(label="Model Dibanding", value="3", delta="TF-IDF, SBERT, IndoBERT")

st.markdown('<p class="footer">Moh. Khairul Umam &middot; NIM 202310370311448 &middot; NLP B &middot; <a href="https://nlp.kair0s.my.id" target="_blank">nlp.kair0s.my.id</a></p>', unsafe_allow_html=True)
