import cv2
import numpy as np

def process_with_otsu(image):
    # Preprocessing
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Segmentasi Otsu
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ekstraksi fitur sederhana (contoh rata-rata intensitas)
    features = np.mean(binary_image)
    return features
