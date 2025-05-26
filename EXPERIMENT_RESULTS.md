# CNN vs ViT Cat Classification Experiment Results

## 🎯 Experiment Overview

This experiment compared the performance of **CNN (ResNet50)** vs **Vision Transformer (ViT)** for binary cat classification using the Oxford-IIIT Pet Dataset.

### Experimental Settings

- **Learning Rate**: 0.001 (CNN), 0.0001 (ViT - automatically adjusted)
- **Batch Size**: 32
- **Epochs**: 10
- **Dataset**: 320 training images, 80 validation images (cats vs dogs)
- **Hardware**: CUDA-enabled GPU

## 📊 Results Summary

### Performance Metrics

| Metric              | CNN (ResNet50) | ViT           | Winner                     |
| ------------------- | -------------- | ------------- | -------------------------- |
| **Accuracy**        | 75.00%         | 96.25%        | ✅ **ViT (+21.25%)**       |
| **Precision**       | 75.00%         | 96.51%        | ✅ **ViT (+21.51%)**       |
| **Recall**          | 75.00%         | 96.25%        | ✅ **ViT (+21.25%)**       |
| **F1-Score**        | 75.00%         | 96.25%        | ✅ **ViT (+21.25%)**       |
| **Inference Speed** | 4.18 ms/image  | 5.31 ms/image | ✅ **CNN (1.13ms faster)** |
| **Model Size**      | 90 MB          | 327 MB        | ✅ **CNN (237MB smaller)** |

### Key Findings

#### 🏆 **Overall Winner: Vision Transformer (ViT)**

**ViT significantly outperformed CNN** with:

- **21.25% higher accuracy** (96.25% vs 75.00%)
- **Superior precision and recall** across both classes
- **Excellent generalization** with minimal overfitting

#### ⚡ **Speed vs Accuracy Trade-off**

- **CNN**: Faster inference (4.18ms) but lower accuracy (75%)
- **ViT**: Slightly slower (5.31ms) but much higher accuracy (96.25%)
- **Trade-off**: +1.13ms inference time for +21.25% accuracy improvement

#### 💾 **Model Size Considerations**

- **CNN**: More compact (90MB) - better for deployment
- **ViT**: Larger (327MB) - requires more storage/memory
- **Trade-off**: +237MB model size for significantly better performance

## 🔍 Detailed Analysis

### Training Behavior

#### CNN (ResNet50)

- **Training Accuracy**: 91.88% (final epoch)
- **Validation Accuracy**: 75.00% (final epoch)
- **Overfitting**: Moderate (16.88% gap between train/val)
- **Convergence**: Stable but plateaued early

#### ViT (Vision Transformer)

- **Training Accuracy**: 100.00% (final epoch)
- **Validation Accuracy**: 96.25% (final epoch)
- **Overfitting**: Minimal (3.75% gap between train/val)
- **Convergence**: Excellent, reached near-perfect performance

### Classification Performance

#### CNN Confusion Matrix

```
              precision    recall  f1-score   support
     Non-Cat       0.75      0.75      0.75        40
         Cat       0.75      0.75      0.75        40
```

#### ViT Confusion Matrix

```
              precision    recall  f1-score   support
     Non-Cat       1.00      0.93      0.96        40
         Cat       0.93      1.00      0.96        40
```

## 🚀 Practical Implications

### When to Use CNN (ResNet50)

- ✅ **Resource-constrained environments** (mobile, edge devices)
- ✅ **Real-time applications** requiring fast inference
- ✅ **Limited storage/memory** scenarios
- ✅ **Good enough accuracy** for the use case

### When to Use ViT

- ✅ **High accuracy requirements** (medical, safety-critical)
- ✅ **Server-side deployment** with adequate resources
- ✅ **Research/academic applications**
- ✅ **When accuracy > speed/size constraints**

## 🛠️ Technical Implementation

### Model Architecture Fixes

- **Fixed ViT head modification**: Used `heads.head` instead of `heads`
- **Optimized learning rates**: CNN (0.001), ViT (0.0001)
- **Proper pre-trained weight loading**: ImageNet weights for both models

### Experiment Infrastructure

- **Separate wandb runs** for direct comparison
- **Automated model saving** with best validation accuracy
- **Comprehensive evaluation metrics** and visualizations
- **Clean experiment management** with batch scripts

## 📈 Visualizations Generated

1. **`cnn_vit_comparison.png`**: Comprehensive 4-panel comparison

   - Performance metrics bar chart
   - Inference speed comparison
   - Model size comparison
   - Radar chart for overall performance

2. **Individual model plots**:
   - Training history curves
   - Confusion matrices
   - Classification reports

## 🎯 Conclusions

1. **ViT demonstrates superior accuracy** for image classification tasks
2. **CNN offers better efficiency** in terms of speed and model size
3. **The choice depends on requirements**: accuracy vs efficiency trade-offs
4. **Both models benefit from proper hyperparameter tuning**
5. **ViT shows better generalization** with less overfitting

## 📁 Saved Models

- **`cnn_best_model.pth`**: Best CNN model (75% accuracy)
- **`vit_best_model.pth`**: Best ViT model (96.25% accuracy)
- **`best_models/`**: Directory with all results and visualizations

## 🔄 Reproducibility

All experiments can be reproduced using:

```bash
# For CNN only
python cat_classifier.py --model cnn --batch-size 32 --epochs 10 --lr 0.001

# For ViT only
python cat_classifier.py --model vit --batch-size 32 --epochs 10 --lr 0.001

# For both models
python cat_classifier.py --model both --batch-size 32 --epochs 10 --lr 0.001
```

---

**Experiment Date**: January 25, 2025  
**Framework**: PyTorch with wandb logging  
**Dataset**: Oxford-IIIT Pet Dataset (Cats vs Dogs subset)
