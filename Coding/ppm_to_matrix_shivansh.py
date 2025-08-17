'''from PIL import Image
import numpy as np
def extract_rgb_matrix(ppm_path):
    img = Image.open(ppm_path).convert('RGB')

    if img.size != (100, 100):
        raise ValueError(f"Image size is {img.size}, expected (100, 100).")
    
    img_array = np.array(img, dtype=np.float32) / 255.0

    if img_array.shape != (100, 100, 3):
        raise ValueError(f"Unexpected shape {img_array.shape}.")
    
    return img_array'''

from PIL import Image
import numpy as np

def extract_rgb_matrix(image_path):
    img = Image.open(image_path).convert("RGB")
    rgb_matrix = np.array(img, dtype=np.float32) / 255.0  # scale 0–1
    return rgb_matrix
