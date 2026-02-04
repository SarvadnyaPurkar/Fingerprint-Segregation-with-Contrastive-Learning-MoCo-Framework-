import torch
import torch.nn as nn
from torchvision import models
import os
from PIL import Image
from torchvision import transforms
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load ResNet50 model
resnet = models.resnet50()

# Path to your custom pretrained weights
weights_path = "fingerprint_moco_weights.pth"  # Replace with your weights path

# Load the state_dict from the pretrained weights with weights_only=True
state_dict = torch.load(weights_path, weights_only=True)

# Exclude 'fc' weights (since they may not match your custom task)
state_dict.pop('fc.weight', None)
state_dict.pop('fc.bias', None)

# Load the rest of the weights into the model (this will skip the 'fc' weights)
resnet.load_state_dict(state_dict, strict=False)

# Modify the fully connected layer (fc) to match your custom task (128 classes for fingerprint identification)
num_classes = 128  # Set to your number of classes (e.g., 128 for fingerprint identification)
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

# Set ResNet model to evaluation mode (no dropout, batchnorm in inference mode)
resnet.eval()

# Define image transformation pipeline (resize, crop, normalize)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to extract features using the modified ResNet model
def extract_features(image_paths, num_samples=5):
    features = []
    labels = []
    
    # Loop through each folder (each representing a person)
    for folder_name in os.listdir(image_paths):
        folder_path = os.path.join(image_paths, folder_name)
        
        # Skip files, only process directories (people's subfolders)
        if os.path.isdir(folder_path):
            # Get all images in the folder (assuming each folder contains 16 augmented images)
            all_images = os.listdir(folder_path)
            
            # Randomly select `num_samples` images from the 16 available images
            selected_images = random.sample(all_images, num_samples)
            
            # Process each selected image
            for image_name in selected_images:
                image_path = os.path.join(folder_path, image_name)
                
                # Open and transform the image
                img = Image.open(image_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
                
                # Forward pass through ResNet to get features
                with torch.no_grad():
                    feature = resnet(img_tensor)  # Get the feature vector from ResNet
                features.append(feature.squeeze().numpy())  # Convert to numpy and add to the list
                labels.append(folder_name)  # Store the folder name as the label (person identifier)
    
    return np.array(features), labels

# Path to the folder containing the augmented images (adjust to your folder path)
image_folder = r"C:\\Users\\sarva\\Downloads\\FingerPrintDataset\\FingerPrintProject\\Final_images"  # Replace with the actual path to your image folder

# Extract features from the images (5 images per person)
features, labels = extract_features(image_folder, num_samples=4)

# Standardize the features (important for PCA)
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Apply PCA to reduce the dimensionality to 2D
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features)

# Create a scatter plot
plt.figure(figsize=(10, 8))

# Use a color map to assign different colors to each person (folder)
unique_labels = np.unique(labels)
colors = plt.cm.get_cmap("tab20", len(unique_labels))  # Color map with enough colors

# Plot the PCA features for each person with a unique color
for idx, label in enumerate(unique_labels):
    indices = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(features_pca[indices, 0], features_pca[indices, 1], 
                label=label, color=colors(idx), s=50, alpha=0.6)

# Add labels and title to the plot
plt.title("PCA of ResNet Features (4 Images per Person)", fontsize=16)
plt.xlabel("PCA Component 1", fontsize=12)
plt.ylabel("PCA Component 2", fontsize=12)
plt.legend(title="Person", bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot with a tight layout
plt.tight_layout()
plt.show()
