from PIL import Image

# Load the WEBP file
webp_file = "sample-digit_3.webp"
png_file = "converted_sample_digit_3.png"

# Open and save as PNG
image = Image.open(webp_file)
image.save(png_file, "PNG")

print(f"File converted and saved as {png_file}")