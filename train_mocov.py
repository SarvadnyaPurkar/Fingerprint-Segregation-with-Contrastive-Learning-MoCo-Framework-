import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet50
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 1. Dataset Preparation
class FingerprintDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        for person_folder in os.listdir(root_dir):
            person_path = os.path.join(root_dir, person_folder)
            if os.path.isdir(person_path):
                images = [os.path.join(person_path, img) for img in os.listdir(person_path)]
                if len(images) == 4:  # Only include folders with exactly 4 images
                    self.data.append(images)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_paths = self.data[idx]
        images = [Image.open(img).convert("RGB") for img in image_paths]

        if self.transform:
            images = [self.transform(img) for img in images]

        # Return all 4 images as a tensor batch
        return torch.stack(images)

# Define transformations
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8,5)),  # Random scaling
    transforms.RandomRotation(degrees=150),               # Random rotation
    transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),  # Random Gaussian blur
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the tensor
])

# Create dataset and dataloader
dataset = FingerprintDataset(root_dir=r"C:\\Users\\sarva\\Downloads\\FingerPrintDataset\\FingerPrintProject\\denoise_images", transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

# 2. MoCo Implementation
class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07):
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T

        # Encoder
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        # Initialize key encoder with weights from query encoder
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, images):
        im_q, im_k = images[:, 0], images[:, 1]  # Use the first two images as query and key

        # Query Encoder
        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            # Momentum update for key encoder
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

            k = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=1)

        # MoCo Loss
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        # Update queue
        batch_size = k.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[:, ptr:ptr + batch_size] = k.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

        return logits, labels

# 3. Training Loop
def train_moco(dataloader, epochs=100, save_path="fingerprint_moco_weights.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoCo(base_encoder=resnet50).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for images in tqdm(dataloader):
            images = images.to(device)  # [batch_size, 4, C, H, W]
            
            logits, labels = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

    torch.save(model.encoder_q.state_dict(), save_path)
    print(f"Weights saved to {save_path}")

# 4. Run Training
if __name__ == "__main__":
    train_moco(dataloader, epochs=100, save_path="fingerprint_moco_weights.pth")
