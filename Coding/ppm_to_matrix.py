from PIL import Image
import numpy as np

def ppm_to_matrix(image_path):

    img = Image.open(image_path)
    
    rgb_matrix = np.array(img)

    return rgb_matrix
