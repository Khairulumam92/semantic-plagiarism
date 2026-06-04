import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Evaluasi - Deteksi Plagiarisme",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .eval-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .eval-sub {
        color: #666;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        border-top: 4px solid #4a90d9;
    }
    .metric-card h4 {
        margin: 0 0 0.5rem 0;
        color: #555;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .metric-card .sub {
        font-size: 0.8rem;
        color: #888;
        margin-top: 0.25rem;
    }
    .winner {
        border-top-color: #51cf66;
        background: linear-gradient(180deg, #f0fff4 0%, white 100%);
    }
    .table-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="eval-header">Evaluasi Model</p>', unsafe_allow_html=True)
st.markdown('<p class="eval-sub">Perbandingan performa tiga pendekatan deteksi plagiarisme semantik</p>', unsafe_allow_html=True)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(base_dir, 'assets', 'hasil_evaluasi.csv')

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)

    best_model = df.loc[df['f1'].idxmax()]
    best_auc = df.loc[df['auc'].idxmax()]

    st.markdown("### Ringkasan Performa")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""
        <div class="metric-card winner">
            <h4>Model Terbaik</h4>
            <div class="value">{best_model['model']}</div>
            <div class="sub">F1-Score tertinggi</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>F1-Score Terbaik</h4>
            <div class="value">{best_model['f1']:.4f}</div>
            <div class="sub">{best_model['model']}</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>AUC-ROC Terbaik</h4>
            <div class="value">{best_auc['auc']:.4f}</div>
            <div class="sub">{best_auc['model']}</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Precision Tertinggi</h4>
            <div class="value">{df['precision'].max():.4f}</div>
            <div class="sub">{df.loc[df['precision'].idxmax(), 'model']}</div>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Dataset</h4>
            <div class="value">1.725</div>
            <div class="sub">pasangan kalimat</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Tabel Perbandingan Lengkap")
    st.markdown('<div class="table-container">', unsafe_allow_html=True)

    df_display = df.copy()
    df_display = df_display.style.format({
        'f1': '{:.4f}',
        'precision': '{:.4f}',
        'recall': '{:.4f}',
        'auc': '{:.4f}',
        'threshold': '{:.4f}'
    })
    st.dataframe(df_display, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Analisis Per Model")

    tab1, tab2, tab3 = st.tabs(["TF-IDF", "SBERT", "IndoBERT"])

    for i, (_, row) in enumerate(df.iterrows()):
        name = row['model']
        with [tab1, tab2, tab3][i]:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**F1-Score:** {row['f1']:.4f}")
                st.markdown(f"**Precision:** {row['precision']:.4f}")
                st.markdown(f"**Recall:** {row['recall']:.4f}")
            with c2:
                st.markdown(f"**AUC-ROC:** {row['auc']:.4f}")
                st.markdown(f"**Threshold:** {row['threshold']:.4f}")

            interpretations = {
                "TF-IDF": "Model baseline yang mengandalkan frekuensi kata. Performa cukup baik untuk teks dengan kemiripan kata langsung, namun kurang mampu menangkap makna semantik yang berbeda secara leksikal.",
                "SBERT": "Model terbaik secara keseluruhan. Embedding multilingual memungkinkan pemahaman konteks yang lebih baik, menghasilkan F1 dan AUC tertinggi di antara ketiga model.",
                "IndoBERT": "Model khusus Bahasa Indonesia dengan arsitektur BERT. Performa sedikit di bawah SBERT karena tidak di-fine-tune untuk task paraphrase detection secara spesifik."
            }
            st.info(interpretations[name], icon="💡")

    st.markdown("---")
    st.markdown("### Visualisasi")

    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        st.markdown("**ROC Curve**")
        roc_path = os.path.join(base_dir, 'assets', 'roc_curve.png')
        if os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True)
        else:
            st.warning("Grafik ROC belum tersedia.")

    with viz_col2:
        st.markdown("**Confusion Matrix**")
        cm_model = st.selectbox("Pilih model", ["tfidf", "sbert", "indobert"], key="cm_select")
        cm_path = os.path.join(base_dir, 'assets', f'confusion_matrix_{cm_model}.png')
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
        else:
            st.warning(f"Confusion matrix untuk {cm_model.upper()} belum tersedia.")

else:
    st.warning("File evaluasi belum tersedia. Jalankan `python run_pipeline.py` terlebih dahulu.", icon="⚠️")
