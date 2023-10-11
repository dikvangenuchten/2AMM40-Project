import os
import random
from PIL import Image, ImageDraw

# Define the size of the images and the background color (white)
image_size = (64, 64)
background_color = 255  # White in 1-bit mode

# Define the list of shapes you want to include (e.g., square and circle)
shapes = ['square', 'circle']

# Define the number of data points you want to generate
num_data_points = 100  # You can change this value as needed

# Define the directory to save the dataset
output_dir = 'image_dataset'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to generate and save a single image
def generate_image(filename):
    image = Image.new('1', image_size, background_color)
    draw = ImageDraw.Draw(image)

    # Randomly choose a shape from the list
    shape = random.choice(shapes)

    # Define the shape parameters (position, size)
    position = (random.randint(0, image_size[0] - 30), random.randint(0, image_size[1] - 30))
    size = random.randint(10, 30)

    # Draw the selected shape on the image (black)
    if shape == 'square':
        draw.rectangle([position, (position[0] + size, position[1] + size)], fill=0)
    elif shape == 'circle':
        draw.ellipse([position, (position[0] + size, position[1] + size)], fill=0)

    # Save the generated image
    image.save(os.path.join(output_dir, filename))

# Generate the specified number of images
for i in range(num_data_points):
    filename = f'image_{i}.png'
    generate_image(filename)

print(f'{num_data_points} images have been generated and saved in the "{output_dir}" directory.')
