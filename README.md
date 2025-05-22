# Cat Classification: CNN vs Vision Transformer Comparison

This project implements and compares two approaches for cat image classification:

1. Convolutional Neural Network (CNN) using ResNet50
2. Vision Transformer (ViT) using ViT-B-16

## Overview

This project demonstrates a comparative analysis of traditional Convolutional Neural Networks (CNN) versus modern Vision Transformers (ViT) for cat image classification. It evaluates performance metrics including accuracy, inference time, and classification quality.

## Project Structure

- `cat_classifier.py`: Main script that implements both CNN and ViT models for training and evaluation
- `download_images.py`: Script to download and organize the Oxford-IIIT Pet Dataset
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
```

## Dataset

The project uses the Oxford-IIIT Pet Dataset, which contains 37 categories of pet images, including various cat and dog breeds. The images are organized into:

- Cats (label 1): Various cat breeds
- Non-cats (label 0): Various dog breeds

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
python cat_classifier.py --epochs 10 --batch-size 16 --output-dir output
```

Parameters:

- `--data-dir`: Path to dataset directory (default: 'cat_dataset')
- `--batch-size`: Batch size for training (default: 16)
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 0.001)
- `--output-dir`: Directory to save results (default: 'output')

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
