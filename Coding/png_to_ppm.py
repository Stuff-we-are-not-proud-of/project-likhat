from PIL import Image

def convert_image_to_ppm(input_image, output_image):
    img = Image.open(input_image)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(output_image, format="PPM")