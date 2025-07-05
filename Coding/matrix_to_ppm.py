import numpy as np

def matrix_to_ppm(matrix, output_ppm_path, width, height):
    try:
        matrix = np.asarray(matrix)
        if matrix.ndim == 2:
            matrix = matrix[:, :, np.newaxis]
        if matrix.shape != (height, width, 1):
            raise ValueError(f"Expected matrix shape ({height}, {width}, 1), got {matrix.shape}")
        
        matrix = (matrix * 255).astype(np.uint8)
        
        rgb_array = np.repeat(matrix, 3, axis=2)
        print("RGB array shape:", rgb_array.shape)
        
        with open(output_ppm_path, 'wb') as f:
            header = f"P6\n{width} {height}\n255\n"
            f.write(header.encode('ascii'))
            
            f.write(rgb_array.tobytes())
        
        print(f"PPM file saved to {output_ppm_path}")
    
    except Exception as e:
        print(f"Error in matrix_to_ppm: {e}")
        raise

input_text_file = r""
dimensions_file = r""
output_ppm_file = r""

try:
    with open(dimensions_file, 'r') as f:
        width, height = map(int, f.read().strip().split())
    print(f"Loaded dimensions: width={width}, height={height}")
    
    matrix = np.loadtxt(input_text_file)
    matrix = matrix[:, :, np.newaxis]
    print("Loaded matrix shape:", matrix.shape)
    
    matrix_to_ppm(matrix, output_ppm_file, width, height)
    
except Exception as e:
    print(f"Error: {e}")