import os
import shutil

# Define the root folder containing the images
base_dir = "data" 
root_folder = os.path.join(base_dir, "denoise_images")

# Get a list of all images in the folder and sort them to ensure the order is maintained
images = sorted(os.listdir(root_folder))

# Create 200 folders and distribute the images
for i in range(200):
    # Create a new subfolder
    subfolder_name = f"Person_{i + 1}"
    subfolder_path = os.path.join(root_folder, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)
    
    # Calculate the range of images for this folder
    start_idx = i * 4
    end_idx = start_idx + 4
    
    # Move the images to the subfolder
    for img in images[start_idx:end_idx]:
        src_path = os.path.join(root_folder, img)
        dest_path = os.path.join(subfolder_path, img)
        shutil.move(src_path, dest_path)

print("Images organized into folders successfully!")


