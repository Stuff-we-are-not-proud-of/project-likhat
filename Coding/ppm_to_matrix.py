import numpy as np

def read_ppm_to_grayscale(file_path):
    try:
        with open(file_path, 'rb') as f:
            header_lines = []
            while len(header_lines) < 3:
                line = f.readline().strip()
                if not line or line.startswith(b'#'):
                    continue
                header_lines.append(line.decode('ascii'))
            
            print("Parsed header lines:", header_lines)
            
            if not header_lines or header_lines[0] != 'P6':
                raise ValueError("Invalid PPM file: Expected P6 format")
            if len(header_lines) < 3:
                raise ValueError(f"Invalid PPM header: Expected 3 lines, got {len(header_lines)}")
            
            try:
                width, height = map(int, header_lines[1].split())
                maxval = int(header_lines[2])
            except (ValueError, IndexError) as e:
                raise ValueError(f"Invalid PPM header: Unable to parse width/height or maxval: {e}")
            
            if maxval != 255:
                raise ValueError(f"Only maxval 255 is supported, got {maxval}")
            
            pixel_data = f.read()
            expected_size = width * height * 3
            print(f"Pixel data size: {len(pixel_data)} bytes, expected: {expected_size} bytes")
            if len(pixel_data) != expected_size:
                raise ValueError(f"Pixel data size ({len(pixel_data)}) does not match expected ({expected_size})")
            
            img_array = np.frombuffer(pixel_data, dtype=np.uint8).reshape(height, width, 3)
            print("RGB array shape:", img_array.shape)
            
            grayscale = (
                0.299 * img_array[:, :, 0] + 
                0.587 * img_array[:, :, 1] + 
                0.114 * img_array[:, :, 2]   
            ).astype(np.float32)
            print("Grayscale 2D shape:", grayscale.shape)
            
            grayscale = grayscale[:, :, np.newaxis] / 255.0
            print("Final grayscale 3D shape:", grayscale.shape)
            
            return grayscale, width, height
    except Exception as e:
        print(f"Error in read_ppm_to_grayscale: {e}")
        raise

file_path = r""
output_text_file = r""
dimensions_file = r""

try:
    img_matrix, width, height = read_ppm_to_grayscale(file_path)
    print("Grayscale matrix shape:", img_matrix.shape)
    
    np.savetxt(output_text_file, img_matrix[:, :, 0], fmt='%.6f')
    print(f"Matrix saved to {output_text_file}")
    
    with open(dimensions_file, 'w') as f:
        f.write(f"{width} {height}")
    print(f"Dimensions saved to {dimensions_file}")
    
except Exception as e:
    print(f"Error: {e}")