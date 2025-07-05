import os
from PIL import Image

def convert_to_ppm(input_path, output_path):
    try:
        img = Image.open(input_path)
<<<<<<< HEAD
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
=======
        img = img.convert('L')  # Convert to grayscale
        img.save(output_path, format='PPM', binary=True)
        print(f"Successfully converted {input_path} to {output_path}")
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def create_ppm_dataset(input_root, output_root):
    # Verify input directory exists
    if not os.path.exists(input_root):
        print(f"Error: Input directory {input_root} does not exist")
        return
    
    # Create output directory
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        print(f"Created output directory: {output_root}")
    
    # Define character groups
    capital_letters = [chr(i) for i in range(65, 91)]  # A-Z
    small_letters = [chr(i) for i in range(97, 123)]   # a-z
    numbers = [str(i) for i in range(10)]              # 0-9
    
    # Create subfolders for CapitalLetters, SmallLetters, Numbers
    for group, chars in [
        ('CapitalLetters', capital_letters),
        ('SmallLetters', small_letters),
        ('Numbers', numbers)
    ]:
        group_dir = os.path.join(output_root, group)
        if not os.path.exists(group_dir):
            os.makedirs(group_dir)
            print(f"Created group directory: {group_dir}")
        for char in chars:
            char_dir = os.path.join(group_dir, char)
            if not os.path.exists(char_dir):
                os.makedirs(char_dir)
                print(f"Created character directory: {char_dir}")
    
    # Process files
    file_count = 0
    success_count = 0
    for root, dirs, files in os.walk(input_root):
        print(f"Scanning directory: {root}")
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_count += 1
                input_path = os.path.join(root, file)
                # Extract character (first part before '_')
                try:
                    char = file.split('_')[0]
                    # Validate character
                    if len(char) == 1:
                        if char in capital_letters:
                            output_dir = os.path.join(output_root, 'CapitalLetters', char)
                        elif char in small_letters:
                            output_dir = os.path.join(output_root, 'SmallLetters', char)
                        elif char in numbers:
                            output_dir = os.path.join(output_root, 'Numbers', char)
                        else:
                            print(f"Skipping {input_path}: Invalid character '{char}'")
                            continue
                        output_filename = os.path.splitext(file)[0] + '.ppm'
                        output_path = os.path.join(output_dir, output_filename)
                        if convert_to_ppm(input_path, output_path):
                            success_count += 1
                    else:
                        print(f"Skipping {input_path}: Character '{char}' is not a single character")
                except IndexError:
                    print(f"Skipping {input_path}: Filename does not contain '_'")
    
    print(f"Processed {file_count} files, successfully converted {success_count} to PPM")
>>>>>>> data-collect

if __name__ == "__main__":
    input_root = os.path.join("MajorDatasets", "AugmentedDataset")
    output_root = os.path.join("MajorDatasets", "PPMFormatDataset")
    create_ppm_dataset(input_root, output_root)