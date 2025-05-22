import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import seaborn as sns
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights
from sklearn.metrics import confusion_matrix, classification_report

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CatDataset(Dataset):
    """Dataset for cat vs non-cat classification."""
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Directory with all the images.
            split (str): 'train' or 'val'.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Paths to cat and non-cat folders
        cat_dir = os.path.join(root_dir, split, 'cat')
        non_cat_dir = os.path.join(root_dir, split, 'non_cat')
        
        # Get all image paths and labels
        self.image_paths = []
        self.labels = []
        
        # Add cat images (label 1)
        for img_name in os.listdir(cat_dir):
            if img_name.endswith('.jpg'):
                self.image_paths.append(os.path.join(cat_dir, img_name))
                self.labels.append(1)
        
        # Add non-cat images (label 0)
        for img_name in os.listdir(non_cat_dir):
            if img_name.endswith('.jpg'):
                self.image_paths.append(os.path.join(non_cat_dir, img_name))
                self.labels.append(0)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder instead of failing
            placeholder = torch.zeros((3, 224, 224)) if self.transform else Image.new('RGB', (224, 224))
            return placeholder, self.labels[idx]

def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=10):
    """Train the model."""
    
    # Initialize tracking variables
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Track statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # Calculate epoch loss and accuracy
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            # Record history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.cpu().numpy())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.cpu().numpy())
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Deep copy the model if best accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        
        print()
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

def evaluate_model(model, dataloader, criterion):
    """Evaluate the model on the test set."""
    model.eval()
    
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []
    
    # Measure inference time
    start_time = time.time()
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Track statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    total_time = time.time() - start_time
    dataset_size = len(dataloader.dataset)
    avg_inference_time = total_time / dataset_size
    test_loss = running_loss / dataset_size
    test_acc = running_corrects.double() / dataset_size
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Create classification report
    report = classification_report(all_labels, all_preds, target_names=['Non-Cat', 'Cat'])
    
    results = {
        'loss': test_loss,
        'accuracy': test_acc.cpu().numpy(),
        'confusion_matrix': cm,
        'classification_report': report,
        'inference_time': avg_inference_time
    }
    
    return results

def plot_training_history(history, title, save_path):
    """Plot training and validation loss/accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Validation')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, class_names, title, save_path):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_comparison_metrics(cnn_results, vit_results, save_path):
    """Plot comparison of metrics between CNN and ViT."""
    metrics = ['Accuracy', 'Inference Time (ms)']
    cnn_values = [
        cnn_results['accuracy'] * 100,  # Convert to percentage
        cnn_results['inference_time'] * 1000  # Convert to ms
    ]
    vit_values = [
        vit_results['accuracy'] * 100,  # Convert to percentage
        vit_results['inference_time'] * 1000  # Convert to ms
    ]
    
    # Create bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy plot
    ax1.bar(['CNN', 'ViT'], [cnn_values[0], vit_values[0]], color=['blue', 'orange'])
    ax1.set_title('Accuracy (%)')
    ax1.set_ylim(0, 100)
    
    # Inference time plot
    ax2.bar(['CNN', 'ViT'], [cnn_values[1], vit_values[1]], color=['blue', 'orange'])
    ax2.set_title('Inference Time (ms per image)')
    
    plt.suptitle('CNN vs ViT Performance Comparison')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train and compare CNN and ViT models for cat classification')
    parser.add_argument('--data-dir', type=str, default='cat_dataset', help='Path to dataset directory')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory for results')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Define transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Create datasets
    train_dataset = CatDataset(args.data_dir, 'train', data_transforms['train'])
    val_dataset = CatDataset(args.data_dir, 'val', data_transforms['val'])
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0),  # Set num_workers to 0 to avoid multiprocessing issues on Windows
        'val': DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    }
    
    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset)
    }
    
    print(f"Dataset sizes: Train={dataset_sizes['train']}, Val={dataset_sizes['val']}")
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Train CNN model (ResNet50)
    print("\n===== Training CNN (ResNet50) Model =====")
    
    # Initialize ResNet50 with pre-trained weights
    cnn_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    
    # Modify final fully connected layer for binary classification
    num_features = cnn_model.fc.in_features
    cnn_model.fc = nn.Linear(num_features, 2)  # 2 classes: cat and non-cat
    cnn_model = cnn_model.to(device)
    
    # Optimizer
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=args.lr)
    
    # Train the CNN model
    cnn_model, cnn_history = train_model(
        cnn_model, criterion, cnn_optimizer, dataloaders, 
        dataset_sizes, num_epochs=args.epochs
    )
    
    # Evaluate CNN model
    print("\nEvaluating CNN model...")
    cnn_results = evaluate_model(cnn_model, dataloaders['val'], criterion)
    
    # Save CNN model
    torch.save(cnn_model.state_dict(), os.path.join(args.output_dir, 'cnn_model.pth'))
    
    # Train Vision Transformer model
    print("\n===== Training Vision Transformer (ViT) Model =====")
    
    # Initialize ViT with pre-trained weights
    vit_model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    
    # Modify the head for binary classification
    vit_model.heads = nn.Linear(vit_model.hidden_dim, 2)  # 2 classes: cat and non-cat
    vit_model = vit_model.to(device)
    
    # Optimizer
    vit_optimizer = optim.Adam(vit_model.parameters(), lr=args.lr)
    
    # Train the ViT model
    vit_model, vit_history = train_model(
        vit_model, criterion, vit_optimizer, dataloaders, 
        dataset_sizes, num_epochs=args.epochs
    )
    
    # Evaluate ViT model
    print("\nEvaluating ViT model...")
    vit_results = evaluate_model(vit_model, dataloaders['val'], criterion)
    
    # Save ViT model
    torch.save(vit_model.state_dict(), os.path.join(args.output_dir, 'vit_model.pth'))
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Plot training history
    plot_training_history(cnn_history, 'CNN Training History', 
                        os.path.join(args.output_dir, 'cnn_history.png'))
    plot_training_history(vit_history, 'ViT Training History', 
                        os.path.join(args.output_dir, 'vit_history.png'))
    
    # Plot confusion matrices
    class_names = ['Non-Cat', 'Cat']
    plot_confusion_matrix(cnn_results['confusion_matrix'], class_names, 
                        'CNN Confusion Matrix', os.path.join(args.output_dir, 'cnn_cm.png'))
    plot_confusion_matrix(vit_results['confusion_matrix'], class_names, 
                        'ViT Confusion Matrix', os.path.join(args.output_dir, 'vit_cm.png'))
    
    # Plot comparison metrics
    plot_comparison_metrics(cnn_results, vit_results, 
                          os.path.join(args.output_dir, 'model_comparison.png'))
    
    # Print results
    print("\n===== Results =====")
    print("\nCNN Results:")
    print(f"Accuracy: {cnn_results['accuracy']*100:.2f}%")
    print(f"Average Inference Time: {cnn_results['inference_time']*1000:.2f} ms per image")
    print("\nClassification Report:")
    print(cnn_results['classification_report'])
    
    print("\nViT Results:")
    print(f"Accuracy: {vit_results['accuracy']*100:.2f}%")
    print(f"Average Inference Time: {vit_results['inference_time']*1000:.2f} ms per image")
    print("\nClassification Report:")
    print(vit_results['classification_report'])
    
    # Create summary report
    with open(os.path.join(args.output_dir, 'summary_report.txt'), 'w') as f:
        f.write("===== Cat Classification: CNN vs ViT Comparison =====\n\n")
        
        f.write("Dataset Information:\n")
        f.write(f"Training samples: {dataset_sizes['train']}\n")
        f.write(f"Validation samples: {dataset_sizes['val']}\n\n")
        
        f.write("CNN (ResNet50) Results:\n")
        f.write(f"Accuracy: {cnn_results['accuracy']*100:.2f}%\n")
        f.write(f"Inference Time: {cnn_results['inference_time']*1000:.2f} ms per image\n\n")
        f.write("Classification Report:\n")
        f.write(cnn_results['classification_report'])
        f.write("\n")
        
        f.write("ViT Results:\n")
        f.write(f"Accuracy: {vit_results['accuracy']*100:.2f}%\n")
        f.write(f"Inference Time: {vit_results['inference_time']*1000:.2f} ms per image\n\n")
        f.write("Classification Report:\n")
        f.write(vit_results['classification_report'])
        f.write("\n")
        
        # Model comparison
        acc_diff = abs(vit_results['accuracy'] - cnn_results['accuracy']) * 100
        time_diff = abs(vit_results['inference_time'] - cnn_results['inference_time']) * 1000
        
        f.write("Model Comparison:\n")
        if vit_results['accuracy'] > cnn_results['accuracy']:
            f.write(f"ViT outperforms CNN in accuracy by {acc_diff:.2f}%\n")
        elif cnn_results['accuracy'] > vit_results['accuracy']:
            f.write(f"CNN outperforms ViT in accuracy by {acc_diff:.2f}%\n")
        else:
            f.write("Both models have equal accuracy\n")
        
        if vit_results['inference_time'] < cnn_results['inference_time']:
            f.write(f"ViT is faster than CNN by {time_diff:.2f} ms per image\n")
        elif cnn_results['inference_time'] < vit_results['inference_time']:
            f.write(f"CNN is faster than ViT by {time_diff:.2f} ms per image\n")
        else:
            f.write("Both models have equal inference time\n")
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == '__main__':
    main() 