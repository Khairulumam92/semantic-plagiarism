import streamlit as st

st.set_page_config(page_title="Deteksi Plagiarisme Semantik", layout="wide")
st.title("Deteksi Plagiarisme Semantik pada Teks Bahasa Indonesia")
st.markdown("""
Aplikasi ini membandingkan tiga pendekatan untuk mendeteksi plagiarisme semantik:
- **TF-IDF** (Baseline klasik)
- **Multilingual Sentence-BERT** (Embedding semantik multilingual)
- **IndoBERT** (Model Bahasa Indonesia)

Silakan navigasi ke halaman:
- **Demo** untuk mencoba model
- **Evaluasi** untuk melihat perbandingan metrik
- **Tentang** untuk penjelasan metode dan dataset
""")
