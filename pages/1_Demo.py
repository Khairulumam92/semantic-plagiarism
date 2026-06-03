import streamlit as st
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import clean_text

st.set_page_config(page_title="Demo", layout="wide")
st.title("Demo Deteksi Plagiarisme")

text1 = st.text_area("Teks 1", height=150)
text2 = st.text_area("Teks 2", height=150)
model_choice = st.selectbox("Pilih Model", ["TF-IDF", "SBERT", "IndoBERT"])

if st.button("Cek Similaritas"):
    if text1.strip() and text2.strip():
        t1 = clean_text(text1)
        t2 = clean_text(text2)

        with st.spinner("Memproses..."):
            if model_choice == "TF-IDF":
                from model_tfidf import TFIDFModel
                model = TFIDFModel()
                model.fit([(t1, t2)])
                score = model.get_similarity(t1, t2)
                threshold = 0.5
            elif model_choice == "SBERT":
                from model_sbert import SBERTModel
                model = SBERTModel()
                score = model.get_similarity(t1, t2)
                threshold = 0.75
            else:
                from model_indobert import IndoBERTModel
                model = IndoBERTModel()
                score = model.get_similarity(t1, t2)
                threshold = 0.75

        is_plagiarized = score >= threshold

        st.metric("Skor Similaritas", f"{score:.4f}")
        if is_plagiarized:
            st.success("TERDETEKSI PLAGIARISME SEMANTIK")
        else:
            st.warning("TIDAK TERDETEKSI PLAGIARISME SEMANTIK")
        st.info(f"Threshold yang digunakan: {threshold}")
    else:
        st.error("Masukkan kedua teks terlebih dahulu.")
