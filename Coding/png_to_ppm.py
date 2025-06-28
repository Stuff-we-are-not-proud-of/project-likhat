from PIL import Image

input_image = r""  # Replace with your image file
output_image = "output.ppm"  

img = Image.open(input_image)
img.save(output_image, format="PPM")