import os
from PIL import Image
from torchvision import transforms
import torch
import pywt
import torch.nn as nn
import torch.nn.functional as F

class MWCNN(nn.Module):
    def __init__(self):
        super(MWCNN, self).__init__()

    def wavelet_denoising(self, img_tensor, wavelet='db1', threshold=0.1):
        img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        coeffs2 = pywt.dwt2(img_np, wavelet)
        LL, (LH, HL, HH) = coeffs2

        # Thresholding for denoising
        LH = pywt.threshold(LH, threshold, mode='soft')
        HL = pywt.threshold(HL, threshold, mode='soft')
        HH = pywt.threshold(HH, threshold, mode='soft')

        denoised_img_np = pywt.idwt2((LL, (LH, HL, HH)), wavelet)
        denoised_img_tensor = torch.from_numpy(denoised_img_np).permute(2, 0, 1).unsqueeze(0).float()

        return denoised_img_tensor

    def forward(self, img_tensor):
        return self.wavelet_denoising(img_tensor)

def denoise_images(input_folder, output_folder, model):
    os.makedirs(output_folder, exist_ok=True)

    # Define image transformations: Convert to tensor and normalize if needed
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Example normalization
    ])

    # Iterate over all images in the input folder
    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)

        if os.path.isfile(img_path) and img_name.lower().endswith(('png', 'jpg', 'jpeg')):  # Check if image file
            # Open and transform the image
            img = Image.open(img_path).convert('RGB')  # Ensure RGB
            img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

            # Apply the denoising model (MWCNN with wavelet denoising)
            with torch.no_grad():  # Disable gradient computation (no need for training)
                denoised_tensor = model(img_tensor)

            # Post-processing: Convert the tensor back to image
            denoised_img = denoised_tensor.squeeze(0)  # Remove batch dimension
            denoised_img = denoised_img.clamp(0, 1)  # Ensure the values are between 0 and 1

            # Convert tensor back to PIL image
            denoised_img = transforms.ToPILImage()(denoised_img)

            # Convert to RGB if the image mode is RGBA (remove alpha channel)
            if denoised_img.mode == 'RGBA':
                denoised_img = denoised_img.convert('RGB')

            # Save the denoised image to the output folder
            output_path = os.path.join(output_folder, img_name)
            denoised_img.save(output_path)
            print(f"Denoised image saved to {output_path}")


def augment_and_save(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor()
    ])

    for idx, image_name in enumerate(os.listdir(input_folder)):
        if image_name.endswith(('jpg', 'jpeg', 'png')):
            img_path = os.path.join(input_folder, image_name)
            img = Image.open(img_path).convert('RGB')

            folder_name = f"image_{idx}"
            folder_path = os.path.join(output_folder, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            img.save(os.path.join(folder_path, f"{folder_name}_original.png"))

            for i in range(3):
                augmented = transform(img)
                augmented = augmented.clamp(0, 1)
                augmented_img = transforms.ToPILImage()(augmented)
                augmented_img.save(os.path.join(folder_path, f"{folder_name}_aug_{i}.png"))
    print(f"Augmentation completed. Augmented images saved to {output_folder}.")

def main():
    raw_dataset_path = r"C:\Users\sarva\Downloads\FingerPrintDataset\FingerPrintProject\raw_images"
    denoised_dataset_path = r"C:\Users\sarva\Downloads\FingerPrintDataset\FingerPrintProject\denoise_images"
    augmented_dataset_path = r"C:\Users\sarva\Downloads\FingerPrintDataset\FingerPrintProject\augmented_images"

    model = MWCNN()
    print("Starting denoising...")
    denoise_images(raw_dataset_path, denoised_dataset_path, model)

    print("Starting augmentation...")
    augment_and_save(denoised_dataset_path, augmented_dataset_path)

if __name__ == "__main__":
    main()
