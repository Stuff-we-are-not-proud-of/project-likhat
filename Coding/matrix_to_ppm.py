import os
import numpy as np
from PIL import Image

def matrix_to_ppm(input_txt_path, output_ppm_path="reconstructed.ppm"):
    try:
        if not os.path.exists(input_txt_path):
            print(f"Error: Text file {input_txt_path} does not exist")
            return False
        
        matrix = np.zeros((100, 100, 3), dtype=np.float32)
        
        with open(input_txt_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines[1:]:
            if line.strip():
                parts = line.strip().split(': ')
                if len(parts) != 2:
                    print(f"Error: Invalid line format: {line.strip()}")
                    return False
                pixel_info, rgb_values = parts
                i, j = map(int, pixel_info.replace('Pixel (', '').replace(')', '').split(', '))
                r, g, b = map(float, rgb_values.strip('[]').split(', '))
                matrix[i, j] = [r, g, b]
        
        if matrix.shape != (100, 100, 3):
            print(f"Error: Parsed matrix shape {matrix.shape}, expected (100, 100, 3)")
            return False
        
        if np.any(matrix < 0) or np.any(matrix > 1):
            print(f"Warning: Matrix contains values outside [0, 1] range")
        
        img_array = (matrix * 255.0).astype(np.uint8)
        
        img = Image.fromarray(img_array, mode='RGB')
        
        if img.size != (100, 100):
            print(f"Error: Image size {img.size}, expected (100, 100)")
            return False
        
        img.save(output_ppm_path, format='PPM', binary=True)
        
        print(f"Successfully converted matrix from {input_txt_path} to PPM at {output_ppm_path}")
        return True
    
    except Exception as e:
        print(f"Error processing {input_txt_path}: {str(e)}")
        return False

if __name__ == "__main__":
    input_txt_path = r""
    output_ppm_path = r""
    success = matrix_to_ppm(input_txt_path, output_ppm_path)
    
    if success:
        print(f"PPM saved to {output_ppm_path}")
    else:
        print("Failed to convert matrix to PPM")