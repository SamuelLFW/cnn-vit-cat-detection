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
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score

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

def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=10, model_type='cnn'):
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
            all_preds = []
            all_labels = []
            
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
                
                # Collect predictions and labels for metrics
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            # Calculate epoch loss and accuracy
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            # Calculate additional metrics
            if len(np.unique(all_labels)) > 1:  # Ensure we have examples of both classes
                precision = precision_score(all_labels, all_preds, average='weighted')
                recall = recall_score(all_labels, all_preds, average='weighted')
                f1 = f1_score(all_labels, all_preds, average='weighted')
            else:
                precision, recall, f1 = 0, 0, 0
                
            # Record history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.cpu().numpy())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.cpu().numpy())
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Log metrics to wandb
            wandb.log({
                f"{model_type}/{phase}/loss": epoch_loss,
                f"{model_type}/{phase}/accuracy": epoch_acc.cpu().numpy(),
                f"{model_type}/{phase}/precision": precision,
                f"{model_type}/{phase}/recall": recall,
                f"{model_type}/{phase}/f1_score": f1,
                "epoch": epoch
            })
            
            # Deep copy the model if best accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                
                # Log best model to wandb
                model_artifact = wandb.Artifact(
                    f"{model_type}_best_model", type="model",
                    description=f"Best {model_type} model with validation accuracy {best_acc:.4f}"
                )
                
                # Save model with error handling
                model_filename = f"{model_type}_best_model.pth"
                try:
                    torch.save(model.state_dict(), model_filename)
                    model_artifact.add_file(model_filename)
                    wandb.log_artifact(model_artifact)
                except Exception as e:
                    print(f"Warning: Could not save model artifact to wandb: {e}")
                    # Still save the model locally
                    torch.save(model.state_dict(), model_filename)
        
        print()
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

def evaluate_model(model, dataloader, criterion, model_type='cnn'):
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
    report = classification_report(all_labels, all_preds, target_names=['Non-Cat', 'Cat'], output_dict=True)
    report_str = classification_report(all_labels, all_preds, target_names=['Non-Cat', 'Cat'])
    
    # Log metrics to wandb
    wandb.log({
        f"{model_type}/test/loss": test_loss,
        f"{model_type}/test/accuracy": test_acc.cpu().numpy(),
        f"{model_type}/test/inference_time": avg_inference_time,
        f"{model_type}/test/precision": report['weighted avg']['precision'],
        f"{model_type}/test/recall": report['weighted avg']['recall'],
        f"{model_type}/test/f1_score": report['weighted avg']['f1-score'],
    })
    
    # Log confusion matrix to wandb
    wandb.log({
        f"{model_type}/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_labels,
            preds=all_preds,
            class_names=['Non-Cat', 'Cat']
        )
    })
    
    results = {
        'loss': test_loss,
        'accuracy': test_acc.cpu().numpy(),
        'confusion_matrix': cm,
        'classification_report': report_str,
        'inference_time': avg_inference_time,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score']
    }
    
    return results

def plot_training_history(history, title, save_path, model_type='cnn'):
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
    
    # Log the plot to wandb
    wandb.log({f"{model_type}/training_history": wandb.Image(save_path)})
    
    plt.close()

def plot_confusion_matrix(cm, class_names, title, save_path, model_type='cnn'):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    
    # Log the plot to wandb
    wandb.log({f"{model_type}/confusion_matrix_plot": wandb.Image(save_path)})
    
    plt.close()

def plot_comparison_metrics(cnn_results, vit_results, save_path):
    """Plot comparison of metrics between CNN and ViT."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Inference Time (ms)']
    cnn_values = [
        cnn_results['accuracy'] * 100,  # Convert to percentage
        cnn_results['precision'] * 100,
        cnn_results['recall'] * 100,
        cnn_results['f1_score'] * 100,
        cnn_results['inference_time'] * 1000  # Convert to ms
    ]
    vit_values = [
        vit_results['accuracy'] * 100,  # Convert to percentage
        vit_results['precision'] * 100,
        vit_results['recall'] * 100,
        vit_results['f1_score'] * 100,
        vit_results['inference_time'] * 1000  # Convert to ms
    ]
    
    # Create comparison table for wandb
    comparison_table = wandb.Table(
        columns=["Metric", "CNN", "ViT", "Difference (CNN-ViT)"]
    )
    
    for i, metric in enumerate(metrics):
        if metric == 'Inference Time (ms)':
            # For inference time, lower is better
            diff = cnn_values[i] - vit_values[i]
            better = "CNN" if diff < 0 else "ViT" if diff > 0 else "Equal"
        else:
            # For other metrics, higher is better
            diff = cnn_values[i] - vit_values[i]
            better = "CNN" if diff > 0 else "ViT" if diff < 0 else "Equal"
        
        comparison_table.add_data(metric, cnn_values[i], vit_values[i], diff)
    
    wandb.log({"model_comparison/table": comparison_table})
    
    # Create bar chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Accuracy, Precision, Recall, F1 plots
    for i, (metric, ax) in enumerate(zip(metrics[:4], axes)):
        ax.bar(['CNN', 'ViT'], [cnn_values[i], vit_values[i]], color=['blue', 'orange'])
        ax.set_title(f'{metric} (%)')
        ax.set_ylim(0, 100)
    
    # Inference time plot (separate because different scale)
    plt.figure(figsize=(6, 5))
    plt.bar(['CNN', 'ViT'], [cnn_values[4], vit_values[4]], color=['blue', 'orange'])
    plt.title('Inference Time (ms per image)')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(save_path), 'inference_time_comparison.png'))
    
    # Log inference time comparison
    wandb.log({"model_comparison/inference_time": wandb.Image(os.path.join(os.path.dirname(save_path), 'inference_time_comparison.png'))})
    
    # Save and log the main comparison plot
    plt.figure(figsize=(12, 8))
    x = np.arange(len(metrics[:4]))
    width = 0.35
    plt.bar(x - width/2, cnn_values[:4], width, label='CNN')
    plt.bar(x + width/2, vit_values[:4], width, label='ViT')
    plt.xlabel('Metrics')
    plt.ylabel('Value (%)')
    plt.title('CNN vs ViT Performance Comparison')
    plt.xticks(x, metrics[:4])
    plt.legend()
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(save_path)
    
    # Log the main comparison plot
    wandb.log({"model_comparison/performance_metrics": wandb.Image(save_path)})
    
    plt.close('all')

def main():
    parser = argparse.ArgumentParser(description='Train and compare CNN and ViT models for cat classification')
    parser.add_argument('--data-dir', type=str, default='cat_dataset', help='Path to dataset directory')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory for results')
    parser.add_argument('--wandb-project', type=str, default='cat-classification', help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='Weights & Biases entity (username or team name)')
    parser.add_argument('--wandb-api-key', type=str, default=None, help='Weights & Biases API key')
    parser.add_argument('--exp-name', type=str, default=None, help='Experiment name for Weights & Biases')
    parser.add_argument('--dataset-size', type=int, default=None, help='Size of dataset used (for experiment tracking)')
    parser.add_argument('--model', type=str, choices=['cnn', 'vit', 'both'], default='both', 
                       help='Which model to train: cnn, vit, or both')
    args = parser.parse_args()
    
    # Set up wandb
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    
    # Generate experiment name if not provided
    if args.exp_name is None:
        model_suffix = f"-{args.model}" if args.model != 'both' else "-comparison"
        args.exp_name = f"cat-classification{model_suffix}-bs{args.batch_size}-e{args.epochs}-lr{args.lr}"
        if args.dataset_size:
            args.exp_name += f"-ds{args.dataset_size}"
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.exp_name,
        config={
            "architecture": args.model.upper() if args.model != 'both' else "CNN vs ViT",
            "dataset": "Oxford-IIIT Pet Dataset (Cats vs Dogs)",
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "dataset_size": args.dataset_size or "unknown",
            "model_type": args.model
        }
    )
    
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
        'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    }
    
    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset)
    }
    
    print(f"Dataset sizes: Train={dataset_sizes['train']}, Val={dataset_sizes['val']}")
    wandb.config.update({
        "train_samples": dataset_sizes['train'],
        "val_samples": dataset_sizes['val']
    })
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize results
    cnn_results = None
    vit_results = None
    cnn_history = None
    vit_history = None
    
    # Train CNN model (ResNet50) if requested
    if args.model in ['cnn', 'both']:
        print("\n===== Training CNN (ResNet50) Model =====")
        
        # Initialize ResNet50 with pre-trained weights
        cnn_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Modify final fully connected layer for binary classification
        num_features = cnn_model.fc.in_features
        cnn_model.fc = nn.Linear(num_features, 2)  # 2 classes: cat and non-cat
        cnn_model = cnn_model.to(device)
        
        # Optimizer
        cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=args.lr)
        
        # Log model architecture to wandb
        wandb.watch(cnn_model, log="all", log_freq=10)
        
        # Train the CNN model
        cnn_model, cnn_history = train_model(
            cnn_model, criterion, cnn_optimizer, dataloaders, 
            dataset_sizes, num_epochs=args.epochs, model_type='cnn'
        )
        
        # Evaluate CNN model
        print("\nEvaluating CNN model...")
        cnn_results = evaluate_model(cnn_model, dataloaders['val'], criterion, model_type='cnn')
        
        # Save CNN model
        torch.save(cnn_model.state_dict(), os.path.join(args.output_dir, 'cnn_model.pth'))
    
    # Train Vision Transformer model if requested
    if args.model in ['vit', 'both']:
        print("\n===== Training Vision Transformer (ViT) Model =====")
        
        # Initialize ViT with pre-trained weights
        vit_model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
        # Modify the head for binary classification - CORRECTED VERSION
        # The ViT has a nested structure: heads.head, not just heads
        original_in_features = vit_model.heads.head.in_features
        vit_model.heads.head = nn.Linear(original_in_features, 2)  # 2 classes: cat and non-cat
        
        vit_model = vit_model.to(device)
        
        # Use a smaller learning rate for ViT (often needs different LR than CNN)
        vit_lr = args.lr * 0.1  # 10x smaller learning rate for ViT
        print(f"Using ViT learning rate: {vit_lr} (CNN uses {args.lr})")
        
        # Optimizer with smaller learning rate
        vit_optimizer = optim.Adam(vit_model.parameters(), lr=vit_lr)
        
        # Log model architecture to wandb
        wandb.watch(vit_model, log="all", log_freq=10)
        
        # Train the ViT model
        vit_model, vit_history = train_model(
            vit_model, criterion, vit_optimizer, dataloaders, 
            dataset_sizes, num_epochs=args.epochs, model_type='vit'
        )
        
        # Evaluate ViT model
        print("\nEvaluating ViT model...")
        vit_results = evaluate_model(vit_model, dataloaders['val'], criterion, model_type='vit')
        
        # Save ViT model
        torch.save(vit_model.state_dict(), os.path.join(args.output_dir, 'vit_model.pth'))
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Plot training history for trained models
    if cnn_history is not None:
        plot_training_history(cnn_history, 'CNN Training History', 
                            os.path.join(args.output_dir, 'cnn_history.png'), model_type='cnn')
    
    if vit_history is not None:
        plot_training_history(vit_history, 'ViT Training History', 
                            os.path.join(args.output_dir, 'vit_history.png'), model_type='vit')
    
    # Plot confusion matrices for trained models
    class_names = ['Non-Cat', 'Cat']
    if cnn_results is not None:
        plot_confusion_matrix(cnn_results['confusion_matrix'], class_names, 
                            'CNN Confusion Matrix', os.path.join(args.output_dir, 'cnn_cm.png'), model_type='cnn')
    
    if vit_results is not None:
        plot_confusion_matrix(vit_results['confusion_matrix'], class_names, 
                            'ViT Confusion Matrix', os.path.join(args.output_dir, 'vit_cm.png'), model_type='vit')
    
    # Plot comparison metrics only if both models were trained
    if cnn_results is not None and vit_results is not None:
        plot_comparison_metrics(cnn_results, vit_results, 
                              os.path.join(args.output_dir, 'model_comparison.png'))
    
    # Print results
    print("\n===== Results =====")
    
    if cnn_results is not None:
        print("\nCNN Results:")
        print(f"Accuracy: {cnn_results['accuracy']*100:.2f}%")
        print(f"Average Inference Time: {cnn_results['inference_time']*1000:.2f} ms per image")
        print("\nClassification Report:")
        print(cnn_results['classification_report'])
    
    if vit_results is not None:
        print("\nViT Results:")
        print(f"Accuracy: {vit_results['accuracy']*100:.2f}%")
        print(f"Average Inference Time: {vit_results['inference_time']*1000:.2f} ms per image")
        print("\nClassification Report:")
        print(vit_results['classification_report'])
    
    # Create summary report
    with open(os.path.join(args.output_dir, 'summary_report.txt'), 'w') as f:
        f.write(f"===== Cat Classification: {args.model.upper()} Results =====\n\n")
        
        f.write("Dataset Information:\n")
        f.write(f"Training samples: {dataset_sizes['train']}\n")
        f.write(f"Validation samples: {dataset_sizes['val']}\n\n")
        
        if cnn_results is not None:
            f.write("CNN (ResNet50) Results:\n")
            f.write(f"Accuracy: {cnn_results['accuracy']*100:.2f}%\n")
            f.write(f"Inference Time: {cnn_results['inference_time']*1000:.2f} ms per image\n\n")
            f.write("Classification Report:\n")
            f.write(cnn_results['classification_report'])
            f.write("\n\n")
        
        if vit_results is not None:
            f.write("ViT Results:\n")
            f.write(f"Accuracy: {vit_results['accuracy']*100:.2f}%\n")
            f.write(f"Inference Time: {vit_results['inference_time']*1000:.2f} ms per image\n\n")
            f.write("Classification Report:\n")
            f.write(vit_results['classification_report'])
            f.write("\n\n")
        
        # Model comparison only if both models were trained
        if cnn_results is not None and vit_results is not None:
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
    
    # Save summary report as artifact
    summary_artifact = wandb.Artifact(
        f"{args.model}_summary_report", type="report",
        description=f"Summary report for {args.model.upper()} model(s)"
    )
    summary_artifact.add_file(os.path.join(args.output_dir, 'summary_report.txt'))
    wandb.log_artifact(summary_artifact)
    
    print(f"\nResults saved to {args.output_dir}")
    
    # Finish wandb run
    wandb.finish()

if __name__ == '__main__':
    main() 