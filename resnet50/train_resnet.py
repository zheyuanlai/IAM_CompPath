import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
import numpy as np
import tifffile

Image.MAX_IMAGE_PIXELS = None

# ---------------------------
# Helper Functions
# ---------------------------
def extract_patches(img_array, mask_array, patch_size=224, threshold=0.5):
    patches = []
    stride = patch_size
    h, w = mask_array.shape

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            mask_patch = mask_array[y:y+patch_size, x:x+patch_size]
            if np.mean(mask_patch > 0) >= threshold:
                img_patch = img_array[y:y+patch_size, x:x+patch_size, :]
                patches.append(img_patch)

    print(f"Extracted {len(patches)} patches of size {patch_size}x{patch_size}")
    return patches

# ---------------------------
# Dataset Definition
# ---------------------------
class WSIMaskedDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, patch_size=224, downscale_factor=0.1):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.patch_size = patch_size
        self.downscale_factor = downscale_factor

        self.data = []
        self.prepare_dataset()

    def prepare_dataset(self):
        mask_paths = glob.glob(os.path.join(self.mask_dir, "*/*.tif"))
        print(f"Found {len(mask_paths)} mask files.")

        for mask_path in mask_paths:
            mask_basename = os.path.basename(mask_path)
            wsi_id = os.path.basename(os.path.dirname(mask_path))
            original_image_path = os.path.join(self.image_dir, f"{wsi_id}.tiff")

            if not os.path.exists(original_image_path):
                print(f"Original image not found for {wsi_id}, skipping.")
                continue

            # Read mask, convert to grayscale
            mask_img = Image.open(mask_path).convert('L')
            mask_array = np.array(mask_img)

            # Read WSI using tifffile
            orig_img = tifffile.imread(original_image_path)

            # Downscale dimensions by downscale_factor
            new_w = int(orig_img.shape[1] * self.downscale_factor)
            new_h = int(orig_img.shape[0] * self.downscale_factor)

            # Resize the mask
            mask_img_small = mask_img.resize((new_w, new_h), Image.NEAREST)
            mask_array_small = np.array(mask_img_small)

            # Resize the original image
            orig_img_pil = Image.fromarray(orig_img)
            orig_img_small_pil = orig_img_pil.resize((new_w, new_h), Image.LANCZOS)
            orig_img_small = np.array(orig_img_small_pil)

            # Check if dimensions match
            if mask_array_small.shape != orig_img_small.shape[:2]:
                print(f"Dimension mismatch for {wsi_id}, skipping.")
                continue

            # Extract patches on the downscaled arrays
            patches = extract_patches(orig_img_small, mask_array_small, self.patch_size)

            # Determine label based on mask filename
            if ("Normal" in mask_basename) or ("Stroma" in mask_basename):
                label = 0
            else:
                label = 1

            for patch in patches:
                self.data.append((patch, label))

            print(f"WSI {wsi_id}: Added {len(patches)} patches with label {label}")

        print(f"Total dataset size: {len(self.data)} patches")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patch_array, label = self.data[idx]
        patch_img = Image.fromarray(patch_array)

        if self.transform:
            patch_img = self.transform(patch_img)

        return patch_img, label

# ---------------------------
# Data Transforms
# ---------------------------
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------------------
# Dataset Preparation
# ---------------------------
image_dir = "/home/users/nus/e1297740/scratch/cdpl_bii/Scanners/KFBio/original/tiff"
mask_dir = "/home/users/nus/e1297740/scratch/cdpl_bii/mask_corrected_registered/original/KFBio"

full_dataset = WSIMaskedDataset(
    image_dir, 
    mask_dir, 
    transform=train_transforms, 
    patch_size=224, 
    downscale_factor=0.1
)

val_ratio = 0.2
val_size = int(len(full_dataset) * val_ratio)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

val_dataset.dataset.transform = val_transforms

print(f"Train dataset size: {train_size} patches")
print(f"Validation dataset size: {val_size} patches")

# ---------------------------
# DataLoaders
# ---------------------------
batch_size = 4

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# ---------------------------
# Model Definition
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ---------------------------
# Training & Validation Loop
# ---------------------------
epochs = 10

for epoch in range(epochs):
    print(f"\nEpoch [{epoch+1}/{epochs}]")
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_acc = 100 * train_correct / train_total
    train_loss /= train_total
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total
    val_loss /= val_total
    print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%")

# ---------------------------
# Save Model
# ---------------------------
torch.save(model.state_dict(), "resnet50_masked_training.pth")
print("\nModel saved as resnet50_masked_training.pth")
