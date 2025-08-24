import cv2
import numpy as np
import os

def add_gaussian_noise(image, mean=0, sigma=25):
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy = cv2.add(image, gauss)
    return noisy

# Directory paths
clean_dir = r"dataset\clean"
noisy_dir = r"dataset\noisy"
os.makedirs(noisy_dir, exist_ok=True)

# Loop through files and subdirectories in the clean_dir
for root, dirs, files in os.walk(clean_dir):
    for filename in files:
        img_path = os.path.join(root, filename)
        
        # Check if it's a valid image file (ensure file extension is valid)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img = cv2.imread(img_path)
            
            if img is not None:
                noisy_img = add_gaussian_noise(img)
                noisy_img_path = os.path.join(noisy_dir, os.path.relpath(img_path, clean_dir))
                noisy_img_dir = os.path.dirname(noisy_img_path)
                os.makedirs(noisy_img_dir, exist_ok=True)  # Ensure the noisy image directory exists
                cv2.imwrite(noisy_img_path, noisy_img)
            else:
                print(f"Warning: Failed to read image: {img_path}")
        else:
            print(f"Skipping non-image file: {filename}")
