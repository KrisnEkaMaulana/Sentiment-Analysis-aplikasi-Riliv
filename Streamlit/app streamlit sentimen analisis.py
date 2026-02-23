import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix

# === Konfigurasi Halaman ===
st.set_page_config(page_title="Prediksi Sentimen", page_icon="💬", layout="wide")

# === Path ke model ===
model_paths = {
    "Random Forest": "/content/drive/MyDrive/Semester 7/Skripsi/Modeling/FIX/rf_tuned_bundle.joblib_fix",
    "Naive Bayes": "/content/drive/MyDrive/Semester 7/Skripsi/Modeling/FIX/nb_tuned_bundle.joblib_fix"
}

# === Fungsi load model ===
@st.cache_resource
def load_model_bundle(path):
    return joblib.load(path)

# === Load semua model ===
models = {}
for name, path in model_paths.items():
    if not os.path.exists(path):
        st.warning(f"❌ Model {name} tidak ditemukan di path: {path}")
    else:
        models[name] = load_model_bundle(path)

# === Mapping label angka ke teks ===
label_map = {
    0: "Negatif",
    1: "Positif",
    2: "Netral"
}

# === UI Utama ===
st.title("💬 Aplikasi Prediksi Sentimen Ulasan")
st.markdown("Gunakan model klasifikasi untuk memprediksi **sentimen ulasan pengguna**, baik satuan maupun dalam jumlah banyak.")

# === Prediksi Single Teks ===
st.subheader("📝 Masukkan teks ulasan:")
text_input = st.text_area("Tulis ulasan di sini...")

if st.button("🔍 Prediksi"):
    if not text_input.strip():
        st.warning("⚠️ Silakan masukkan teks ulasan.")
    else:
        for model_name, bundle in models.items():
            model = bundle['model']
            vectorizer = bundle['vectorizer']
            X_vec = vectorizer.transform([text_input])
            pred = model.predict(X_vec)[0]
            label = label_map.get(pred, str(pred))

            st.markdown(f"### 🔎 {model_name}")
            st.success(f"✅ Prediksi: **{label}**")
            if label.lower() == "negatif":
                st.markdown("⚠️ *Sentimen negatif. Perlu perhatian lebih lanjut.*")
            elif label.lower() == "positif":
                st.markdown("🎉 *Sentimen positif! Bagus untuk perkembangan aplikasi!*")
            else:
                st.markdown("ℹ️ *Sentimen netral atau tidak terdeteksi.*")

# === Evaluasi Model ===
with st.expander("📈 Lihat Hasil Evaluasi Model"):
    for model_name, bundle in models.items():
        st.markdown(f"### 📊 {model_name}")
        if 'classification_report' in bundle and 'y_true' in bundle and 'y_pred' in bundle:
            report = pd.DataFrame(bundle['classification_report']).transpose()
            st.dataframe(report.style.format(precision=2))

            model = bundle['model']
            cm = confusion_matrix(bundle['y_true'], bundle['y_pred'])
            labels = model.classes_ if hasattr(model, 'classes_') else sorted(set(bundle['y_true']))
            label_names = [label_map.get(l, l) for l in labels]

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=label_names, yticklabels=label_names)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
        else:
            st.info("Evaluasi belum tersedia untuk model ini.")

# === Upload File CSV untuk Prediksi Massal ===
st.markdown("---")
st.subheader("📤 Upload File CSV untuk Prediksi Massal")

uploaded_file = st.file_uploader("Unggah file CSV (harus ada kolom `ulasan`):", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'ulasan' not in df.columns:
        st.error("⚠️ File harus mengandung kolom `ulasan`.")
    else:
        st.success(f"✅ Berhasil membaca {len(df)} ulasan.")
        for model_name, bundle in models.items():
            model = bundle['model']
            vectorizer = bundle['vectorizer']
            X_vec = vectorizer.transform(df['ulasan'].astype(str))
            preds = model.predict(X_vec)
            pred_labels = [label_map.get(p, str(p)) for p in preds]
            df[f"pred_{model_name.lower().replace(' ', '_')}"] = pred_labels

        st.write("📋 Hasil Prediksi:")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Hasil Prediksi CSV",
            data=csv,
            file_name="hasil_prediksi_sentimen.csv",
            mime="text/csv"
        )