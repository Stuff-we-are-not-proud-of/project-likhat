import os
from PIL import Image

def convert_to_ppm(input_path, output_path):
    try:
        img = Image.open(input_path)
        img = img.convert('L')
        img.save(output_path, format='PPM', binary=True)
        print(f"Converted {input_path} to {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def create_ppm_dataset(input_root, output_root):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    
    characters = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)] + [str(i) for i in range(10)]
    for char in characters:
        char_dir = os.path.join(output_root, char)
        if not os.path.exists(char_dir):
            os.makedirs(char_dir)
    
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file)
                char = file.split('_')[0]
                if char in characters:
                    output_dir = os.path.join(output_root, char)
                    output_filename = os.path.splitext(file)[0] + '.ppm'
                    output_path = os.path.join(output_dir, output_filename)
                    try:
                        convert_to_ppm(input_path, output_path)
                    except Exception as e:
                        print(f"Error processing {input_path}: {e}")
                else:
                    print(f"Skipping {input_path}: Invalid character '{char}'")

if __name__ == "__main__":
    input_root = os.path.join("MajorDatasets", "AugmentedDataset")
    output_root = os.path.join("MajorDatasets", "PPMFormatDataset")
    create_ppm_dataset(input_root, output_root)