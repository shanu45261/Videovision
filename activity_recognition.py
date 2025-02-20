# activity_recognition.py
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from PIL import Image
from typing import List, Tuple, Dict
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define activity categories
ACTIVITY_CATEGORIES = {
    'RunningTreadmill': 0,
    'Walking': 1,
    'JumpingJack': 2,
    'Biking': 3,
    'BodyWeightSquats': 4,
    'PushUps': 5,
    'BoxingPunchingBag': 6,
    'BreastStroke': 7,
    'YoYo': 8
}

class VideoDataset(Dataset):
    """Dataset for loading video clips."""
    def __init__(self, root_dir: str, clip_length: int = 16, transform=None):
        self.root_dir = root_dir
        self.clip_length = clip_length
        self.transform = transform
        self.samples = []
        
        # Collect all video paths and their labels
        for activity in os.listdir(root_dir):
            if activity in ACTIVITY_CATEGORIES:
                activity_path = os.path.join(root_dir, activity)
                for video_name in os.listdir(activity_path):
                    if video_name.endswith(('.avi', '.mp4', '.mov')):
                        video_path = os.path.join(activity_path, video_name)
                        self.samples.append((video_path, ACTIVITY_CATEGORIES[activity]))
    
    def _load_video(self, video_path: str) -> torch.Tensor:
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        # Calculate frame sampling rate
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise ValueError(f"No frames found in video: {video_path}")
            
        sampling_rate = max(1, total_frames // self.clip_length)
        
        frame_count = 0
        while len(frames) < self.clip_length:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sampling_rate == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
                
            frame_count += 1
        
        cap.release()
        
        # If we don't have enough frames, loop the video
        while len(frames) < self.clip_length:
            frames.extend(frames[:self.clip_length-len(frames)])
        
        # Take exactly clip_length frames
        frames = frames[:self.clip_length]
        return torch.stack(frames)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, label = self.samples[idx]
        try:
            clip = self._load_video(video_path)
            return clip, label
        except Exception as e:
            print(f"Error loading video {video_path}: {str(e)}")
            # Return a random different sample
            return self.__getitem__((idx + 1) % len(self))

class ActivityNet(nn.Module):
    """3D CNN architecture for activity recognition."""
    def __init__(self, num_classes: int):
        super(ActivityNet, self).__init__()
        
        # Load pretrained R3D_18 model
        self.backbone = models.video.r3d_18(pretrained=True)
        
        # Modify final classification layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, frames, channels, height, width)
        # Reshape for 3D CNN: (batch, channels, frames, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        return self.backbone(x)

def train_epoch(model: nn.Module,
                dataloader: DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    for clips, labels in progress_bar:
        clips, labels = clips.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{running_loss/total:.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(dataloader), 100. * correct / total

def validate(model: nn.Module,
            dataloader: DataLoader,
            criterion: nn.Module,
            device: torch.device) -> Tuple[float, float]:
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validating')
        for clips, labels in progress_bar:
            clips, labels = clips.to(device), labels.to(device)
            
            outputs = model(clips)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{running_loss/total:.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    return running_loss / len(dataloader), 100. * correct / total

def plot_training_history(history: Dict):
    """Plot training and validation metrics."""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Training settings
    BATCH_SIZE = 8
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001
    CLIP_LENGTH = 16
    
    # Paths
    TRAIN_PATH = 'datasets/train'
    VAL_PATH = 'datasets/val'
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((112, 112)),  # Smaller size for 3D CNN
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = VideoDataset(TRAIN_PATH, CLIP_LENGTH, transform)
    val_dataset = VideoDataset(VAL_PATH, CLIP_LENGTH, transform)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Initialize model
    print("Initializing model...")
    model = ActivityNet(num_classes=len(ACTIVITY_CATEGORIES)).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    print("Starting training...")
    best_val_acc = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_activity_model.pth')
            print(f"Saved new best model with validation accuracy: {val_acc:.2f}%")
        
        # Plot training history
        plot_training_history(history)
    
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == '__main__':
    main()