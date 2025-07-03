import os
from resize import resize_to_100x100

def create_resized_dataset(input_root, output_root):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    
    for root, dirs, files in os.walk(input_root):
        rel_path = os.path.relpath(root, input_root)
        output_dir = os.path.join(output_root, rel_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file)
                output_filename = os.path.splitext(file)[0] + '.png'
                output_path = os.path.join(output_dir, output_filename)
                try:
                    resize_to_100x100(input_path, output_path, format='PNG')
                    print(f"Processed {input_path} to {output_path}")
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    input_root = os.path.join("MajorDatasets", "Dataset")
    output_root = os.path.join("MajorDatasets", "ResizedDataset")
    create_resized_dataset(input_root, output_root)