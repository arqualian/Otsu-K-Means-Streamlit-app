import streamlit as st
import pandas as pd
import numpy as np
import pickle
from EkstraksiFituOtsur import EkstrakFiturOtsu
from EkstraksiFiturKmeans import EkstrakFiturKMeans
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
import pickle
import os

class Backpro:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        self.W1 = np.random.randn(hidden_size1, input_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((hidden_size1, 1))
        self.W2 = np.random.randn(hidden_size2, hidden_size1) * np.sqrt(2. / hidden_size1)
        self.b2 = np.zeros((hidden_size2, 1))
        self.W3 = np.random.randn(output_size, hidden_size2) * np.sqrt(2. / hidden_size2)
        self.b3 = np.zeros((output_size, 1))

    def ReLU(self, Z):
        return np.maximum(Z, 0)

    def softmax(self, Z):
        A = np.exp(Z - np.max(Z)) / np.sum(np.exp(Z - np.max(Z)), axis=0, keepdims=True)
        return A

    def forward_prop(self, X):
        self.Z1 = self.W1.dot(X) + self.b1
        self.A1 = self.ReLU(self.Z1)
        self.Z2 = self.W2.dot(self.A1) + self.b2
        self.A2 = self.ReLU(self.Z2)
        self.Z3 = self.W3.dot(self.A2) + self.b3
        self.A3 = self.softmax(self.Z3)
        return self.A3

    def get_predictions(self):
        return np.argmax(self.A3, axis=0)

    def make_predictions(self, X):
        self.forward_prop(X)
        return self.get_predictions()

    @staticmethod
    def load_model(file_path):
        with open(file_path, 'rb') as f:
            model_params = pickle.load(f)
        input_size = model_params['W1'].shape[1]
        hidden_size1 = model_params['W1'].shape[0]
        hidden_size2 = model_params['W2'].shape[0]
        output_size = model_params['W3'].shape[0]
        model = Backpro(input_size, hidden_size1, hidden_size2, output_size)
        model.W1 = model_params['W1']
        model.b1 = model_params['b1']
        model.W2 = model_params['W2']
        model.b2 = model_params['b2']
        model.W3 = model_params['W3']
        model.b3 = model_params['b3']
        return model
    
labels = ['Avicennia alba', 'Bruguiera cylindrica', 'Scaevola taccada', 'Scyphiphora hydrophyllacea']

# Fungsi untuk memuat file dengan cache
@st.cache_resource
def load_file(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

# Fungsi untuk memproses gambar dan ekstraksi fitur
def process_image(image_path, method="otsu"):
    if method == "otsu":
        features = EkstrakFiturOtsu(image_path)
    elif method == "kmeans":
        features = EkstrakFiturKMeans(image_path)
    else:
        raise ValueError("Invalid method. Choose 'otsu' or 'kmeans'.")

    if features is None:
        st.error("Ekstraksi fitur gagal. Pastikan gambar dapat diproses.")
        return None

    new_data = np.array([[
        features['Hu_Moment_1'], features['Hu_Moment_2'], features['Hu_Moment_3'],
        features['Hu_Moment_4'], features['Hu_Moment_5'], features['Hu_Moment_6'], features['Hu_Moment_7'],
        features['Perimeter'], features['Diameter'], features['Area'],
        features['ASM_0'], features['ASM_45'], features['ASM_90'], features['ASM_135'],
        features['Contrast_0'], features['Contrast_45'], features['Contrast_90'], features['Contrast_135'],
        features['IDM_0'], features['IDM_45'], features['IDM_90'], features['IDM_135'],
        features['Entropy_0'], features['Entropy_45'], features['Entropy_90'], features['Entropy_135'],
        features['Correlation_0'], features['Correlation_45'], features['Correlation_90'], features['Correlation_135']
    ]])
    return new_data

def main():
    st.title("Mangrove Image Segmentation and Identification App")

    # Pilihan model
    model_type = st.sidebar.selectbox("Pilih Model", ["Backpro Otsu", "Backpro K-Means"])
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    # Tentukan jalur file untuk model, scaler, dan encoder
    model_paths = {
        "Backpro Otsu": {
            "scaler": r"Scaler and Lab Encoder\Otsuscaler.pkl",
            "encoder": r"Scaler and Lab Encoder\Otsulabel_encoder.pkl",
            "model": r"backproOtsu_model.pkl"
        },
        "Backpro K-Means": {
            "scaler": r"Scaler and Lab Encoder\Kmeansscaler.pkl",
            "encoder": r"Scaler and Lab Encoder\Kmeanslabel_encoder.pkl",
            "model": r"backproKMeans_model.pkl"
        }
    }

    paths = model_paths[model_type]
    scaler = load_file(paths["scaler"])
    label_encoder = load_file(paths["encoder"])
    model = Backpro.load_model(paths["model"])

    if uploaded_file is not None:
        try:
            # Tampilkan gambar yang diunggah
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)  # Perbaiki parameter

            # Simpan file sementara
            temp_file_path = "temp_image.png"
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(uploaded_file.getbuffer())

            # Ekstraksi fitur berdasarkan metode yang dipilih
            method = "otsu" if model_type == "Backpro Otsu" else "kmeans"
            features = process_image(temp_file_path, method=method)  # Pastikan fungsi ini digunakan

            if features is not None:
                    # Ubah ke format yang sesuai dengan model
                    scaled_features = scaler.transform(features)

                    # Prediksi hasil
                    predicted_class = model.make_predictions(scaled_features.T)

                    # Tampilkan nama jenis mangrove
                    st.success(labels[int(predicted_class[0])])
                    
            else:
                st.error("Ekstraksi fitur gagal. Pastikan gambar sesuai dengan format yang diterima.")

            # Hapus file sementara
            os.remove(temp_file_path)

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    main()
