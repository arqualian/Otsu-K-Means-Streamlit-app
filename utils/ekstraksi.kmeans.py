import cv2
import numpy as np

def process_with_kmeans(image):
    # Preprocessing
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Segmentasi K-Means
    pixel_values = gray_image.reshape((-1, 1)).astype('float32')
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(gray_image.shape).astype('uint8')

    # Ekstraksi fitur sederhana (contoh rata-rata intensitas)
    features = np.mean(segmented_image)
    return features
