import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ------------------------------
# Data Augmentation
# ------------------------------
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.RandomAffine(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dir = "dataset/train"
val_dir = "dataset/val"
test_dir = "dataset/test"

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)

class_names = train_dataset.classes
print("Classes:", class_names)

# ------------------------------
# Balanced Sampling
# ------------------------------
labels = np.array(train_dataset.targets)
class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
samples_weights = class_weights[labels]
sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ------------------------------
# Model - Gradual Unfreezing
# ------------------------------
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # Freeze backbone

num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 128),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(128, len(class_names))
)

model.to(device)

criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
optimizer = optim.Adam(model.fc.parameters(), lr=0.0008)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# ------------------------------
# Training Loop with Early Stopping
# ------------------------------
best_acc = 0
patience = 5 
patience_count = 0

for epoch in range(30):
    model.train()
    running_loss = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/30"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"\nValidation Accuracy: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_resnet_accident_model.pth")
        patience_count = 0
        print("Model improved and saved!")
    else:
        patience_count += 1
        if patience_count >= patience:
            print("Early stopping triggered.")
            break

print("Training complete. Best Val Accuracy:", best_acc)
