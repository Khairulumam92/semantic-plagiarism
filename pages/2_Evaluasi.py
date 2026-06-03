import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Evaluasi", layout="wide")
st.title("Evaluasi Model")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(base_dir, 'assets', 'hasil_evaluasi.csv')

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    st.dataframe(df, use_container_width=True)
else:
    st.warning("File evaluasi belum tersedia. Jalankan notebook `03_evaluation_final.ipynb` terlebih dahulu.")

st.subheader("ROC Curve")
roc_path = os.path.join(base_dir, 'assets', 'roc_curve.png')
if os.path.exists(roc_path):
    st.image(roc_path)
else:
    st.warning("Grafik ROC belum tersedia.")

st.subheader("Confusion Matrix")
cm_dir = os.path.join(base_dir, 'assets')
for model in ['tfidf', 'sbert', 'indobert']:
    cm_path = os.path.join(cm_dir, f'confusion_matrix_{model}.png')
    if os.path.exists(cm_path):
        st.image(cm_path, caption=f"Confusion Matrix - {model.upper()}")
