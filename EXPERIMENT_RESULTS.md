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
| **Accuracy**        | 82.5%          | 95%           | ✅ **ViT (+12.5%)**        |
| **Precision**       | 87.03%         | 95%           | ✅ **ViT (+7.97%)**        |
| **Recall**          | 82.5%          | 95%           | ✅ **ViT (+12.5%)**        |
| **F1-Score**        | 82.40%         | 95%           | ✅ **ViT (+12.60%)**       |
| **Inference Speed** | 4.18 ms/image  | 5.31 ms/image | ✅ **CNN (1.13ms faster)** |
| **Model Size**      | 90 MB          | 327 MB        | ✅ **CNN (237MB smaller)** |

### Key Findings

#### 🏆 **Overall Winner: Vision Transformer (ViT)**

**ViT significantly outperformed CNN** with:

- **12.5% higher accuracy** (95% vs 82.5%)
- **Superior precision and recall** across both classes
- **Excellent generalization** with minimal overfitting

#### ⚡ **Speed vs Accuracy Trade-off**

- **CNN**: Faster inference (4.18ms) but lower accuracy (82.5%)
- **ViT**: Slightly slower (5.31ms) but much higher accuracy (95%)
- **Trade-off**: +1.13ms inference time for +12.5% accuracy improvement

#### 💾 **Model Size Considerations**

- **CNN**: More compact (90MB) - better for deployment
- **ViT**: Larger (327MB) - requires more storage/memory
- **Trade-off**: +237MB model size for significantly better performance

## 🔍 Detailed Analysis

### Training Behavior

#### CNN (ResNet50)

- **Training Accuracy**: 91.88% (final epoch)
- **Validation Accuracy**: 82.5% (final epoch)
- **Overfitting**: Moderate (9.38% gap between train/val)
- **Convergence**: Stable but plateaued early

#### ViT (Vision Transformer)

- **Training Accuracy**: 100.00% (final epoch)
- **Validation Accuracy**: 95.00% (final epoch)
- **Overfitting**: Minimal (5.00% gap between train/val)
- **Convergence**: Excellent, reached near-perfect performance

### Classification Performance

#### CNN Confusion Matrix

```
              precision    recall  f1-score   support
     Non-Cat       0.87      0.80      0.83        40
         Cat       0.87      0.85      0.86        40
```

#### ViT Confusion Matrix

```
              precision    recall  f1-score   support
     Non-Cat       0.95      0.93      0.94        40
         Cat       0.93      0.95      0.94        40
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

- **`cnn_best_model.pth`**: Best CNN model (82.5% accuracy)
- **`vit_best_model.pth`**: Best ViT model (95% accuracy)
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
