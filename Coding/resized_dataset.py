import os
from PIL import Image

def resize_to_100x100_png(input_path, output_path, target_size=(100, 100)):
    try:
        img = Image.open(input_path)
        img = img.convert('L')  # Convert to grayscale
        orig_width, orig_height = img.size
        max_dim = max(orig_width, orig_height)
        padded_img = Image.new('L', (max_dim, max_dim), color=255)  # White background
        offset_x = (max_dim - orig_width) // 2
        offset_y = (max_dim - orig_height) // 2
        padded_img.paste(img, (offset_x, offset_y))
        resized_img = padded_img.resize(target_size, Image.LANCZOS)
        resized_img.save(output_path, format='PNG')
        print(f"Saved resized image to {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

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
                    resize_to_100x100_png(input_path, output_path)
                    print(f"Processed {input_path} to {output_path}")
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    input_root = os.path.join("MajorDatasets", "Dataset")
    output_root = os.path.join("MajorDatasets", "ResizedDataset")
    create_resized_dataset(input_root, output_root)