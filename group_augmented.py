import os
import shutil

# Define the root folder containing the 800 subfolders
root_folder = r"C:\\Users\\sarva\\Downloads\\FingerPrintDataset\\FingerPrintProject\\augmented_images"

# Get a list of all subfolders and sort them
subfolders = sorted([f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))])

# Create combined folders
combined_folder_count = len(subfolders) // 4  # Since 4 consecutive folders are combined
for i in range(combined_folder_count):
    # Create the new folder for combined contents
    combined_folder_name = f"Combined_{i + 1}"
    combined_folder_path = os.path.join(root_folder, combined_folder_name)
    os.makedirs(combined_folder_path, exist_ok=True)
    
    # Identify the 4 consecutive folders to combine
    start_idx = i * 4
    end_idx = start_idx + 4
    folders_to_combine = subfolders[start_idx:end_idx]
    
    # Move the contents of these folders to the combined folder
    for folder in folders_to_combine:
        folder_path = os.path.join(root_folder, folder)
        for img in os.listdir(folder_path):
            img_src_path = os.path.join(folder_path, img)
            img_dest_path = os.path.join(combined_folder_path, img)
            shutil.move(img_src_path, img_dest_path)

        # Optionally, remove the empty folder after moving its contents
        os.rmdir(folder_path)

print("Folders combined successfully!")
