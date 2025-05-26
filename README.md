# Cat Classification: CNN vs Vision Transformer Comparison

This project implements and compares two approaches for cat image classification:

1. Convolutional Neural Network (CNN) using ResNet50
2. Vision Transformer (ViT) using ViT-B-16

## Overview

This project demonstrates a comparative analysis of traditional Convolutional Neural Networks (CNN) versus modern Vision Transformers (ViT) for cat image classification. It evaluates performance metrics including accuracy, inference time, and classification quality.

## Project Structure

- `cat_classifier.py`: Main script that implements both CNN and ViT models for training and evaluation
- `download_images.py`: Script to download and organize the Oxford-IIIT Pet Dataset
- `run_experiments.ps1`: PowerShell script to run multiple experiment configurations (Windows)
- `run_experiments.sh`: Bash script to run multiple experiment configurations (Linux/Mac)
- `requirements.txt`: Required dependencies
- `cat_dataset/`: Contains the training and validation images
- `output/`: Contains trained models, visualizations, and comparison reports

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.6.12
matplotlib>=3.5.0
numpy>=1.20.0
Pillow>=9.0.0
scikit-learn>=1.0.0
tqdm>=4.60.0
requests>=2.25.0
wandb>=0.15.0
seaborn>=0.12.0
```

## Dataset

The project uses the Oxford-IIIT Pet Dataset, which contains 37 categories of pet images, including various cat and dog breeds. The images are organized into:

- Cats (label 1): Various cat breeds
- Non-cats (label 0): Various dog breeds

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your Weights & Biases API key:
   ```
   WANDB_API_KEY=your_api_key_here
   ```

## Usage

### Download Dataset

```bash
python download_images.py --output_dir cat_dataset --cat_limit 100 --non_cat_limit 100
```

Parameters:

- `--output_dir`: Directory to save the dataset (default: 'cat_dataset')
- `--cat_limit`: Maximum number of cat images to use (default: 100)
- `--non_cat_limit`: Maximum number of non-cat images to use (default: 100)

### Training and Comparison

```bash
python cat_classifier.py --epochs 10 --batch-size 16 --output-dir output --wandb-project "cat-classification" --exp-name "my-experiment"
```

Parameters:

- `--data-dir`: Path to dataset directory (default: 'cat_dataset')
- `--batch-size`: Batch size for training (default: 16)
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 0.001)
- `--output-dir`: Directory to save results (default: 'output')
- `--wandb-project`: Weights & Biases project name (default: 'cat-classification')
- `--wandb-entity`: Weights & Biases entity/username (optional)
- `--wandb-api-key`: Weights & Biases API key (can be provided here or in .env file)
- `--exp-name`: Experiment name for Weights & Biases (optional)
- `--dataset-size`: Size of dataset used, for experiment tracking (optional)

### Running Multiple Experiments

#### Windows (PowerShell)

```powershell
.\run_experiments.ps1
```

#### Linux/Mac (Bash)

```bash
bash run_experiments.sh
```

These scripts will:

1. Read the WANDB_API_KEY from your .env file
2. Run experiments with various configurations (dataset sizes, epochs, learning rates)
3. Show progress bars for tracking experiment completion
4. Save all results to the experiments directory and Weights & Biases

You can modify the configuration parameters in the scripts to adjust the experiments.

### Storage Optimization

Both experiment scripts include storage optimization features to handle disk space efficiently:

```powershell
# Storage optimization settings (PowerShell)
$STORAGE_OPTIMIZATION = $true # Set to false to keep all datasets
$USE_SHARED_DATASET = $true # Set to true to use a single shared dataset for all experiments
$SHARED_DATASET_SIZE = 100 # Number of images per class for the shared dataset
$CLEANUP_AFTER_SIZE_CHANGE = $true # Clean up previous dataset when changing sizes
$CLEANUP_AFTER_ALL = $true # Clean up all datasets after all experiments
```

```bash
# Storage optimization settings (Bash)
STORAGE_OPTIMIZATION=true # Set to false to keep all datasets
USE_SHARED_DATASET=true # Set to true to use a single shared dataset for all experiments
SHARED_DATASET_SIZE=100 # Number of images per class for the shared dataset
CLEANUP_AFTER_SIZE_CHANGE=true # Clean up previous dataset when changing sizes
CLEANUP_AFTER_ALL=true # Clean up all datasets after all experiments
```

These settings provide several options to manage storage:

1. **Shared Dataset Mode**: Uses a single dataset for all experiments (default: enabled)
2. **Progressive Cleanup**: Removes the previous dataset when moving to a new dataset size (default: enabled)
3. **Final Cleanup**: Removes all datasets after all experiments are complete (default: enabled)
4. **Emergency Cleanup**: Automatically cleans up temporary files when disk space is low (<5GB)

You can adjust these settings at the top of the experiment scripts to suit your storage needs.

## Performance Metrics

The comparison includes:

- Accuracy, precision, recall, and F1 score
- Inference time per image
- Training and validation loss curves
- Confusion matrices for both models

## Model Details

### CNN (ResNet50)

- Pre-trained on ImageNet
- Fine-tuned for binary cat classification
- Modified final layer for 2-class output

### Vision Transformer (ViT-B-16)

- Pre-trained on ImageNet
- Fine-tuned for binary cat classification
- Patch size: 16x16
- Input resolution: 224x224
- Modified head for 2-class output

## Results

See the generated `summary_report.txt` in the output directory for detailed performance metrics and model comparison.

For a comprehensive view of all experiments, visit your Weights & Biases dashboard, where you can:

- Compare model performance across different configurations
- View interactive visualizations of training metrics
- Analyze confusion matrices and model predictions
- Track resource usage and inference time
