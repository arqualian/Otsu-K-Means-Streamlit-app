import cv2
import numpy as np
from PIL import Image
import os
from skimage.feature import graycomatrix, graycoprops

def preproses_image(img, size=(225, 225)):
    """Resize the image, apply grayscale, CLAHE, and dilation."""
    # Resize the image
    img_resized = img.resize(size, Image.Resampling.LANCZOS)
    
    # Convert the image to numpy array
    img_array = np.array(img_resized)
    
    # Check if image is already in grayscale (single channel)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # Convert to grayscale if image is in RGB (3 channels)
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        # If image is already grayscale, use it as is
        img_gray = img_array

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)

    # Apply Dilation (using elliptical structuring element)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_dilated = cv2.dilate(img_clahe, kernel, iterations=2)
    
    return img_dilated

def process_image_for_KMeans(img_dilated, k=2, padding=10):
    """Process the image using K-Means clustering and contour detection, keeping only the largest object."""

    # Set pixels with intensity 0-5 to 0
    image = np.where(img_dilated <= 5, 0, img_dilated)

    # Add padding around the image
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)

    # Step 2: Apply Smoothing (2D Convolution)
    kernel = np.ones((5, 5), np.float32) / 25
    smoothed_image = cv2.filter2D(padded_image, -1, kernel)

    # Step 3: Reshape image for K-Means
    pixel_values = smoothed_image.reshape((-1, 1))  # Assuming single channel grayscale
    pixel_values = np.float32(pixel_values)

    # Step 4: Define K-Means criteria and apply
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Step 5: Convert centers to uint8 and recreate segmented image
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(smoothed_image.shape)

    # Step 6: Detect Contours on Segmented Image
    _, binary_segmented = cv2.threshold(segmented_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Keep only the largest contour
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Create a mask with only the largest object
        largest_object_mask = np.zeros_like(binary_segmented)
        cv2.drawContours(largest_object_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # Mask the original image with the largest object
        grayscale_masked = cv2.bitwise_and(padded_image, padded_image, mask=largest_object_mask)

        return largest_object_mask, grayscale_masked

    else:
        print("No contours found.")
        return None, None

def extract_glcm_features(grayscale_image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """Extract GLCM (Gray-Level Co-occurrence Matrix) features from an image."""
    
    # Compute GLCM
    glcm = graycomatrix(grayscale_image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

    # Compute GLCM properties for each angle
    asm = graycoprops(glcm, 'ASM').flatten()
    contrast = graycoprops(glcm, 'contrast').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()

    # Calculate entropy manually for each angle
    entropy = []
    for i in range(len(angles)):
        glcm_angle = glcm[:, :, 0, i]
        entropy.append(-np.sum(glcm_angle * np.log2(glcm_angle + (glcm_angle == 0))))

    # Return all features as a dictionary
    return {
        "ASM": asm,
        "Contrast": contrast,
        "Homogeneity": homogeneity,
        "Correlation": correlation,
        "Entropy": entropy
    }


def extract_shape_features(binary_image):
    """Extract shape features including Hu Moments, Diameter, Perimeter, and Area."""
    
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Use the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate features
    moments = cv2.moments(largest_contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    perimeter = cv2.arcLength(largest_contour, True)
    area = cv2.contourArea(largest_contour)
    diameter = 2 * np.sqrt(area / np.pi) if area > 0 else 0

    # Return individual shape features
    return hu_moments, perimeter, diameter, area

def EkstrakFiturKMeans(image_path, padding=10, size=(225, 225)):
    """Process the image and extract both shape and texture features."""
    
    # Load the image
    img = Image.open(image_path)

    # Preprocess the image (resize, grayscale, CLAHE, dilation)
    img_dilated = preproses_image(img, size)

    # Process the image using Otsu thresholding and get the largest object mask and grayscale masked image
    binary_mask, grayscale_masked = process_image_for_otsu(img_dilated, padding)

    if binary_mask is None or grayscale_masked is None:
        print("No valid contours found.")
        return None

    # Extract GLCM features from the grayscale image (after Otsu)
    glcm_features = extract_glcm_features(grayscale_masked)

    # Extract shape features from the binary image (after Otsu)
    hu_moments, perimeter, diameter, area = extract_shape_features(binary_mask)

    # Return all extracted features separately
    return {
        "ASM_0": glcm_features["ASM"][0], "ASM_45": glcm_features["ASM"][1], 
        "ASM_90": glcm_features["ASM"][2], "ASM_135": glcm_features["ASM"][3],
        "Contrast_0": glcm_features["Contrast"][0], "Contrast_45": glcm_features["Contrast"][1], 
        "Contrast_90": glcm_features["Contrast"][2], "Contrast_135": glcm_features["Contrast"][3],
        "IDM_0": glcm_features["Homogeneity"][0], "IDM_45": glcm_features["Homogeneity"][1], 
        "IDM_90": glcm_features["Homogeneity"][2], "IDM_135": glcm_features["Homogeneity"][3],
        "Entropy_0": glcm_features["Entropy"][0], "Entropy_45": glcm_features["Entropy"][1], 
        "Entropy_90": glcm_features["Entropy"][2], "Entropy_135": glcm_features["Entropy"][3],
        "Correlation_0": glcm_features["Correlation"][0], "Correlation_45": glcm_features["Correlation"][1], 
        "Correlation_90": glcm_features["Correlation"][2], "Correlation_135": glcm_features["Correlation"][3],
        "Hu_Moment_1": hu_moments[0],
        "Hu_Moment_2": hu_moments[1],
        "Hu_Moment_3": hu_moments[2],
        "Hu_Moment_4": hu_moments[3],
        "Hu_Moment_5": hu_moments[4],
        "Hu_Moment_6": hu_moments[5],
        "Hu_Moment_7": hu_moments[6],
        "Perimeter": perimeter,
        "Diameter": diameter,
        "Area": area
    }
