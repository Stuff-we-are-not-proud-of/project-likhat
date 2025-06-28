from PIL import Image

def resize_to_100x100(input_path, output_path, target_size=(100, 100)):
    try:
        img = Image.open(input_path)
        
        img = img.convert('RGB')
        
        img = img.convert('L') 
        
        orig_width, orig_height = img.size
        
        max_dim = max(orig_width, orig_height)
        padded_img = Image.new('L', (max_dim, max_dim), color=255) 
        offset_x = (max_dim - orig_width) // 2
        offset_y = (max_dim - orig_height) // 2
        padded_img.paste(img, (offset_x, offset_y))
        
        resized_img = padded_img.resize(target_size, Image.LANCZOS)
        
        resized_img.save(output_path, format='PPM', binary=True)
        print(f"Saved resized image to {output_path}")
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

