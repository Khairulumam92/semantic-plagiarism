import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Evaluasi - Deteksi Plagiarisme",
    page_icon="📊",
    layout="wide"
)

# Load CSS
css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'styles.css')
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Page Header
st.markdown("""
<div class="page-header">
    <h1>📊 Evaluasi Model</h1>
    <p>Perbandingan performa tiga pendekatan deteksi plagiarisme semantik</p>
</div>
""", unsafe_allow_html=True)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(base_dir, 'assets', 'hasil_evaluasi.csv')

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)

    best_model = df.loc[df['f1'].idxmax()]
    best_auc = df.loc[df['auc'].idxmax()]

    # Summary Stats
    st.markdown("## Ringkasan Performa")

    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        st.markdown(f"""
        <div class="metric-card metric-card-winner animate-fadeIn">
            <div class="metric-card-label">Model Terbaik</div>
            <div class="metric-card-value">{best_model['model']}</div>
            <div class="metric-card-sub">F1-Score tertinggi</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card animate-fadeIn">
            <div class="metric-card-label">F1-Score Terbaik</div>
            <div class="metric-card-value">{best_model['f1']:.4f}</div>
            <div class="metric-card-sub">{best_model['model']}</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card animate-fadeIn">
            <div class="metric-card-label">AUC-ROC Terbaik</div>
            <div class="metric-card-value">{best_auc['auc']:.4f}</div>
            <div class="metric-card-sub">{best_auc['model']}</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card animate-fadeIn">
            <div class="metric-card-label">Precision Tertinggi</div>
            <div class="metric-card-value">{df['precision'].max():.4f}</div>
            <div class="metric-card-sub">{df.loc[df['precision'].idxmax(), 'model']}</div>
        </div>
        """, unsafe_allow_html=True)

    with m5:
        st.markdown("""
        <div class="metric-card animate-fadeIn">
            <div class="metric-card-label">Dataset</div>
            <div class="metric-card-value">1.725</div>
            <div class="metric-card-sub">Pasangan kalimat</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Comparison Table
    st.markdown("## Tabel Perbandingan Lengkap")

    df_display = df.copy()
    df_display['model'] = df_display['model'].apply(lambda x: f"**{x}**")
    
    st.dataframe(
        df_display.style.format({
            'f1': '{:.4f}',
            'precision': '{:.4f}',
            'recall': '{:.4f}',
            'auc': '{:.4f}',
            'threshold': '{:.4f}'
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # Model Analysis
    st.markdown("## Analisis Per Model")

    tab1, tab2, tab3 = st.tabs(["⚙️ TF-IDF", "🌐 SBERT", "🇮🇩 IndoBERT"])

    interpretations = {
        "TF-IDF": {
            "text": "Model baseline yang mengandalkan frekuensi kata. Performa cukup baik untuk teks dengan kemiripan kata langsung, namun kurang mampu menangkap makna semantik yang berbeda secara leksikal.",
            "badge": "model-badge-tfidf"
        },
        "SBERT": {
            "text": "Model terbaik secara keseluruhan. Embedding multilingual memungkinkan pemahaman konteks yang lebih baik, menghasilkan F1 dan AUC tertinggi di antara ketiga model.",
            "badge": "model-badge-sbert"
        },
        "IndoBERT": {
            "text": "Model khusus Bahasa Indonesia dengan arsitektur BERT. Performa sedikit di bawah SBERT karena tidak di-fine-tune untuk task paraphrase detection secara spesifik.",
            "badge": "model-badge-indobert"
        }
    }

    for i, (_, row) in enumerate(df.iterrows()):
        name = row['model']
        with [tab1, tab2, tab3][i]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="info-box info-box-info">
                    <span class="info-box-icon">ℹ️</span>
                    <span><strong>F1-Score:</strong> {row['f1']:.4f} · <strong>Precision:</strong> {row['precision']:.4f}</span>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="info-box info-box-info">
                    <span class="info-box-icon">ℹ️</span>
                    <span><strong>Recall:</strong> {row['recall']:.4f} · <strong>Threshold:</strong> {row['threshold']:.4f}</span>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="info-box info-box-info">
                    <span class="info-box-icon">ℹ️</span>
                    <span><strong>AUC-ROC:</strong> {row['auc']:.4f}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="info-box info-box-success">
                <span class="info-box-icon">💡</span>
                <span>{interpretations[name]['text']}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Visualizations
    st.markdown("## Visualisasi")

    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.markdown("""
        <div class="section-header">
            <span class="section-header-icon">📈</span>
            <h2>ROC Curve</h2>
        </div>
        """, unsafe_allow_html=True)
        
        roc_path = os.path.join(base_dir, 'assets', 'roc_curve.png')
        if os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True)
        else:
            st.markdown("""
            <div class="info-box info-box-warning">
                <span class="info-box-icon">⚠️</span>
                <span>Grafik ROC belum tersedia.</span>
            </div>
            """, unsafe_allow_html=True)

    with viz_col2:
        st.markdown("""
        <div class="section-header">
            <span class="section-header-icon">📊</span>
            <h2>Confusion Matrix</h2>
        </div>
        """, unsafe_allow_html=True)
        
        cm_model = st.selectbox("Pilih model", ["tfidf", "sbert", "indobert"], key="cm_select")
        cm_path = os.path.join(base_dir, 'assets', f'confusion_matrix_{cm_model}.png')
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
        else:
            st.markdown(f"""
            <div class="info-box info-box-warning">
                <span class="info-box-icon">⚠️</span>
                <span>Confusion matrix untuk {cm_model.upper()} belum tersedia.</span>
            </div>
            """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="info-box info-box-warning">
        <span class="info-box-icon">⚠️</span>
        <span>File evaluasi belum tersedia. Jalankan <code>python run_pipeline.py</code> terlebih dahulu.</span>
    </div>
    """, unsafe_allow_html=True)
