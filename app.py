import streamlit as st
from ekstraksi.otsu import process_with_otsu
from ekstraksi.kmeans import process_with_kmeans
import pickle
from PIL import Image

# Muat model
with open('models/model_otsu.pkl', 'rb') as f:
    model_otsu = pickle.load(f)

with open('models/model_kmeans.pkl', 'rb') as f:
    model_kmeans = pickle.load(f)

st.title("Segmentasi dan Prediksi Daun Mangrove")

# Upload gambar
uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Baca gambar
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar Asli", use_column_width=True)

    # Pilih metode segmentasi
    method = st.radio("Pilih Metode Segmentasi", ["Otsu Thresholding", "K-Means Clustering"])

    if method == "Otsu Thresholding":
        # Proses dengan Otsu
        features = process_with_otsu(image)
        prediksi = model_otsu.predict([features])
        st.write(f"Hasil Prediksi (Otsu): {prediksi[0]}")

    elif method == "K-Means Clustering":
        # Proses dengan K-Means
        features = process_with_kmeans(image)
        prediksi = model_kmeans.predict([features])
        st.write(f"Hasil Prediksi (K-Means): {prediksi[0]}")
