import os
import numpy as np
from PIL import Image

def extract_and_save_rgb_matrix(ppm_path, output_txt_path="output_matrix.txt"):
    try:
        if not os.path.exists(ppm_path):
            print(f"Error: PPM file {ppm_path} does not exist")
            return False
        if not ppm_path.lower().endswith('.ppm'):
            print(f"Error: File {ppm_path} is not a PPM file")
            return False
        
        img = Image.open(ppm_path).convert('RGB')
        
        if img.size != (100, 100):
            print(f"Error: Image {ppm_path} is {img.size}, expected (100, 100)")
            return False
        
        img_array = np.array(img) / 255.0
        
        if img_array.shape != (100, 100, 3):
            print(f"Error: Unexpected shape {img_array.shape} for {ppm_path}")
            return False
        
        with open(output_txt_path, 'w') as f:
            f.write("100x100x3 RGB Matrix (normalized to [0, 1])\n")
            for i in range(100):
                for j in range(100):
                    r, g, b = img_array[i, j]
                    f.write(f"Pixel ({i}, {j}): [{r:.6f}, {g:.6f}, {b:.6f}]\n")
        
        print(f"Successfully saved RGB matrix from {ppm_path} to {output_txt_path}")
        return True
    
    except Exception as e:
        print(f"Error processing {ppm_path}: {str(e)}")
        return False

if __name__ == "__main__":
    ppm_path = r""
    output_txt_path = "output_matrix.txt"
    success = extract_and_save_rgb_matrix(ppm_path, output_txt_path)
    
    if success:
        print(f"Matrix saved to {output_txt_path}")
    else:
        print("Failed to extract and save matrix")