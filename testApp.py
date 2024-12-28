import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from EkstraksiFituOtsur import EkstrakFiturOtsu  # Sesuaikan nama file modul ekstraksi fitur Otsu
from EkstraksiFiturKmeans import EkstrakFiturKMeans  # Sesuaikan nama file modul ekstraksi fitur K-Means
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error

# Class untuk implementasi Backpropagation
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

# Fungsi untuk memuat model atau file lainnya dengan cache
@st.cache_resource
def load_file(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

# Fungsi untuk memproses gambar dan ekstraksi fitur
def process_image(image_path, method="otsu"):
    """Process each image and extract features using the specified method."""
    if method == "otsu":
        features = EkstrakFiturOtsu(image_path)
    elif method == "kmeans":
        features = EkstrakFiturKMeans(image_path)
    else:
        raise ValueError("Invalid method. Choose 'otsu' or 'kmeans'.")

    # Periksa jika EkstrakFitur mengembalikan None
    if features is None:
        return None

    # Unpack fitur jika valid
    hu1 = features['Hu_Moment_1']
    hu2 = features['Hu_Moment_2']
    hu3 = features['Hu_Moment_3']
    hu4 = features['Hu_Moment_4']
    hu5 = features['Hu_Moment_5']
    hu6 = features['Hu_Moment_6']
    hu7 = features['Hu_Moment_7']
    perimeter = features['Perimeter']
    diameter = features['Diameter']
    area = features['Area']

    asm_0 = features['ASM_0']
    asm_45 = features['ASM_45']
    asm_90 = features['ASM_90']
    asm_135 = features['ASM_135']
    contrast_0 = features['Contrast_0']
    contrast_45 = features['Contrast_45']
    contrast_90 = features['Contrast_90']
    contrast_135 = features['Contrast_135']
    idm_0 = features['IDM_0']
    idm_45 = features['IDM_45']
    idm_90 = features['IDM_90']
    idm_135 = features['IDM_135']
    entropy_0 = features['Entropy_0']
    entropy_45 = features['Entropy_45']
    entropy_90 = features['Entropy_90']
    entropy_135 = features['Entropy_135']
    correlation_0 = features['Correlation_0']
    correlation_45 = features['Correlation_45']
    correlation_90 = features['Correlation_90']
    correlation_135 = features['Correlation_135']

    # Gabungkan data ke array baru untuk prediksi
    new_data = np.array([[  
        hu1, hu2, hu3, hu4, hu5, hu6, hu7,
        perimeter, diameter, area,
        asm_0, asm_45, asm_90, asm_135,
        contrast_0, contrast_45, contrast_90, contrast_135,
        idm_0, idm_45, idm_90, idm_135,
        entropy_0, entropy_45, entropy_90, entropy_135,
        correlation_0, correlation_45, correlation_90, correlation_135
    ]])
    return new_data

def main():
    st.title("Mangrove Identification App")

    # Pilihan model
    model_type = st.sidebar.selectbox("Pilih Model", ["Backpro Otsu", "Backpro K-Means"])
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    # Tentukan jalur file untuk model, scaler, dan encoder
    model_paths = {
        "Backpro Otsu": {
            "scaler": r"C:\\Users\\USER\\Desktop\\Selected Perogram\\Steamlit Mangrove Segmentation\\Scaler and Lab Encoder\\Otsuscaler.pkl",
            "encoder": r"C:\\Users\\USER\\Desktop\\Selected Perogram\\Steamlit Mangrove Segmentation\\Scaler and Lab Encoder\\Otsulabel_encoder.pkl",
            "model": r"C:\\Users\\USER\\Desktop\\Selected Perogram\\Steamlit Mangrove Segmentation\\backproOtsu_model.pkl"
        },
        "Backpro K-Means": {
            "scaler": r"C:\\Users\\USER\\Desktop\\Selected Perogram\\Steamlit Mangrove Segmentation\\Scaler and Lab Encoder\\Kmeansscaler.pkl",
            "encoder": r"C:\\Users\\USER\\Desktop\\Selected Perogram\\Steamlit Mangrove Segmentation\\Scaler and Lab Encoder\\Kmeanslabel_encoder.pkl",
            "model": r"C:\\Users\\USER\\Desktop\\Selected Perogram\\Steamlit Mangrove Segmentation\\backproKMeans_model.pkl"
        }
    }

    paths = model_paths[model_type]
    scaler = load_file(paths["scaler"])
    label_encoder = load_file(paths["encoder"])
    model = load_file(paths["model"])

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
                predicted_label = label_encoder.inverse_transform([predicted_class[0]])

                # Tampilkan hasil prediksi
                st.success(f"Predicted Label: {predicted_label[0]}")

            else:
                st.error("Ekstraksi fitur gagal. Pastikan gambar sesuai dengan format yang diterima.")

            # Hapus file sementara
            os.remove(temp_file_path)

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")



if __name__ == "__main__":
    main()
