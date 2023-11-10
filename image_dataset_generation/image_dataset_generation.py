import os
import random
from PIL import Image, ImageDraw

# Define the size of the images and the background color (white)
image_size = (64, 64)
background_color = 255  # White in 1-bit mode

# Define the list of shapes you want to include (e.g., square and circle)
shapes = ["square", "circle"]

# Define the number of data points you want to generate for each shape
num_data_points = 100  # You can change this value as needed

# Define the base directory to save the dataset
base_output_dir = "image_dataset"

# Define the proportion of images for the training set (0.8 = 80% for training)
proportion = 0.8

# Create the output directory if it doesn't exist
if not os.path.exists(base_output_dir):
    os.makedirs(base_output_dir)


# Function to generate and save a single image
def generate_image(shape, filename):
    image = Image.new("1", image_size, background_color)
    draw = ImageDraw.Draw(image)

    # Define the shape parameters (position, size)
    position = (
        random.randint(0, image_size[0] - 30),
        random.randint(0, image_size[1] - 30),
    )
    size = random.randint(10, 30)

    # Draw the selected shape on the image (black)
    if shape == "square":
        draw.rectangle([position, (position[0] + size, position[1] + size)], fill=0)
    elif shape == "circle":
        draw.ellipse([position, (position[0] + size, position[1] + size)], fill=0)

    # Determine whether to save the image in the training or test set
    if random.random() < proportion:
        dataset_dir = "train"
    else:
        dataset_dir = "test"

    # Create a subdirectory for the shape if it doesn't exist
    shape_output_dir = os.path.join(base_output_dir, dataset_dir, shape)
    if not os.path.exists(shape_output_dir):
        os.makedirs(shape_output_dir)

    # Save the generated image in the appropriate subdirectory
    image.save(os.path.join(shape_output_dir, filename))


# Generate the specified number of images for each shape
for shape in shapes:
    for i in range(num_data_points):
        filename = f"image_{i}.png"
        generate_image(shape, filename)

print(
    f'{proportion*100}% of the images for each shape have been generated for training and test sets in separate folders in the "{base_output_dir}" directory.'
)
