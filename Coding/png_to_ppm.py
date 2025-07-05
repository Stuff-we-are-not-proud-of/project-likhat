import os
from PIL import Image

def convert_to_ppm(input_path, output_path):
    try:
        if not os.path.exists(input_path):
            print(f"Error: Input file {input_path} does not exist")
            return False
        img = Image.open(input_path)
        img = img.convert('L')
        img.save(output_path, format='PPM', binary=True)
        if os.path.exists(output_path):
            print(f"Successfully converted {input_path} to {output_path}")
            return True
        else:
            print(f"Error: Failed to create file at {output_path}")
            return False
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def create_ppm_dataset(input_root, output_root):
    input_root = os.path.abspath(input_root)
    output_root = os.path.abspath(output_root)
    
    if not os.path.exists(input_root):
        print(f"Error: Input directory {input_root} does not exist")
        return
    print(f"Input directory exists: {input_root}")
    
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        print(f"Created output directory: {output_root}")
    
    group_mapping = {
        'Capital Letters': ([chr(i) for i in range(65, 91)], 'CapitalLetters'),
        'Small Letters': ([chr(i) for i in range(97, 123)], 'SmallLetters'),
        'Numbers': ([str(i) for i in range(10)], 'Numbers')
    }
    
    for input_group, (chars, output_group) in group_mapping.items():
        group_dir = os.path.join(output_root, output_group)
        if not os.path.exists(group_dir):
            os.makedirs(group_dir)
            print(f"Created group directory: {group_dir}")
        for char in chars:
            char_dir = os.path.join(group_dir, char)
            if not os.path.exists(char_dir):
                os.makedirs(char_dir)
                print(f"Created character directory: {char_dir}")
    
    all_characters = [c for chars, _ in group_mapping.values() for c in chars]
    char_counts = {char: 0 for char in all_characters}
    file_count = 0
    success_count = 0
    person_folders = 0
    
    for person_dir in os.listdir(input_root):
        person_path = os.path.join(input_root, person_dir)
        if not os.path.isdir(person_path):
            continue
        person_folders += 1
        print(f"Scanning person directory: {person_path}")
        
        for input_group, (chars, output_group) in group_mapping.items():
            group_path = os.path.join(person_path, input_group)
            if not os.path.exists(group_path):
                print(f"Warning: Group directory {group_path} does not exist")
                continue
            print(f"Scanning group directory: {group_path}")
            group_files = 0
            
            for file in os.listdir(group_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_count += 1
                    group_files += 1
                    input_path = os.path.join(group_path, file)
                    try:
                        base, _ = os.path.splitext(file)
                        parts = base.split('_')
                        char = parts[0]
                        if char not in chars:
                            print(f"Skipping {input_path}: Invalid character '{char}' for group {input_group}")
                            continue
                        
                        if len(parts) >= 2:
                            if parts[1].startswith('aug'):
                                index = parts[1].replace('aug', '')
                                if not index.isdigit():
                                    print(f"Skipping {input_path}: Invalid index '{index}'")
                                    continue
                                output_filename = f"{person_dir}_{char}_aug_{index}.ppm"
                            elif parts[1] == 'orig':
                                output_filename = f"{person_dir}_{char}_orig.ppm"
                            else:
                                print(f"Skipping {input_path}: Second part '{parts[1]}' does not match 'aug' or 'orig'")
                                continue
                        else:
                            print(f"Skipping {input_path}: Filename does not match '<character>_aug<index>' or '<character>_orig'")
                            continue
                        
                        output_dir = os.path.join(output_root, output_group, char)
                        output_path = os.path.join(output_dir, output_filename)
                        if convert_to_ppm(input_path, output_path):
                            success_count += 1
                            char_counts[char] += 1
                        else:
                            print(f"Failed to convert {input_path}")
                    except Exception as e:
                        print(f"Skipping {input_path}: Error parsing filename: {str(e)}")
            print(f"Found {group_files} files in {group_path} (expected ~{len(chars) * 6})")
    
    print(f"Processed {file_count} files across {person_folders} person folders")
    print(f"Successfully converted {success_count} files to PPM")
    expected_augmentations = 6 
    expected_files = 27 * expected_augmentations * 62
    print(f"Expected {expected_files} files (27 people * {expected_augmentations} images * 62 characters)")
    for char in sorted(all_characters):
        expected_char_files = 27 * expected_augmentations
        print(f"Character {char}: {char_counts[char]} images (expected {expected_char_files})")

if __name__ == "__main__":
    input_root = os.path.join("MajorDatasets", "AugmentedDataset")
    output_root = os.path.join("MajorDatasets", "PPMFormatDataset")
    create_ppm_dataset(input_root, output_root)