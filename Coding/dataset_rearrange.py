import os
import shutil
import random
import numpy as np

def create_mini_model_variant(input_root, output_root):
    """
    Reorganize PPMFormatDataset into MiniModelVariant for 62 binary classification models.
    
    Args:
        input_root (str): Path to PPMFormatDataset (e.g., MajorDatasets/PPMFormatDataset).
        output_root (str): Path to MiniModelVariant (e.g., MajorDatasets/MiniModelVariant).
    """
    try:
        # Define character mappings
        capital_letters = [chr(i) for i in range(65, 91)]  # A-Z
        small_letters = [chr(i) for i in range(97, 123)]   # a-z
        numbers = [str(i) for i in range(10)]              # 0-9
        all_characters = capital_letters + small_letters + numbers
        group_mapping = {
            'CapitalLetters': capital_letters,
            'SmallLetters': small_letters,
            'Numbers': numbers
        }
        
        # Create output directory
        output_root = os.path.abspath(output_root)
        if not os.path.exists(output_root):
            os.makedirs(output_root)
            print(f"Created output directory: {output_root}")
        
        # Verify input directory
        input_root = os.path.abspath(input_root)
        if not os.path.exists(input_root):
            print(f"Error: Input directory {input_root} does not exist")
            return
        
        # Create 62 model folders (Model1 to Model62)
        for model_idx, char in enumerate(all_characters, 1):
            model_dir = os.path.join(output_root, f"Model{model_idx}")
            char_dir = os.path.join(model_dir, char)
            not_char_dir = os.path.join(model_dir, f"Not{char}")
            
            # Create directories
            os.makedirs(char_dir, exist_ok=True)
            os.makedirs(not_char_dir, exist_ok=True)
            print(f"Created directories: {char_dir}, {not_char_dir}")
            
            # Copy character images
            # Find source folder (CapitalLetters, SmallLetters, or Numbers)
            source_group = None
            for group, chars in group_mapping.items():
                if char in chars:
                    source_group = group
                    break
            
            if not source_group:
                print(f"Error: Character {char} not found in any group")
                continue
            
            source_dir = os.path.join(input_root, source_group, char)
            if not os.path.exists(source_dir):
                print(f"Error: Source directory {source_dir} does not exist")
                continue
            
            # Copy all PPMs from source_dir to char_dir
            ppm_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.ppm')]
            if len(ppm_files) != 162:
                print(f"Warning: Found {len(ppm_files)} images in {source_dir}, expected 162")
            
            for ppm in ppm_files:
                src_path = os.path.join(source_dir, ppm)
                dst_path = os.path.join(char_dir, ppm)
                shutil.copy2(src_path, dst_path)
            
            print(f"Copied {len(ppm_files)} images to {char_dir}")
            
            # Randomly sample 162 images for Not<character>
            other_chars = [c for c in all_characters if c != char]
            other_images = []
            for other_char in other_chars:
                for group, chars in group_mapping.items():
                    if other_char in chars:
                        other_dir = os.path.join(input_root, group, other_char)
                        if os.path.exists(other_dir):
                            other_ppm_files = [os.path.join(other_dir, f) for f in os.listdir(other_dir) if f.lower().endswith('.ppm')]
                            other_images.extend(other_ppm_files)
            
            if len(other_images) < 162:
                print(f"Error: Not enough images ({len(other_images)}) for Not{char}, need 162")
                continue
            
            # Randomly sample 162 images
            sampled_images = np.random.choice(other_images, size=162, replace=False)
            
            # Copy sampled images to Not<character>
            for idx, src_path in enumerate(sampled_images):
                dst_filename = f"not_{char}_{idx+1}.ppm"
                dst_path = os.path.join(not_char_dir, dst_filename)
                shutil.copy2(src_path, dst_path)
            
            print(f"Copied 162 sampled images to {not_char_dir}")
        
        print(f"Completed: Created 62 model folders in {output_root}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    input_root = r"C:\Users\Krishna Gera\Desktop\Project Likhat\MajorDatasets\PPMFormatDataset"
    output_root = r"C:\Users\Krishna Gera\Desktop\Project Likhat\MajorDatasets\MiniModelVariant"
    create_mini_model_variant(input_root, output_root)