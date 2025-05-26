#!/usr/bin/env python3
"""
Create a comparison chart between CNN and ViT performance.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def create_cnn_vit_comparison():
    """Create a comprehensive comparison chart between CNN and ViT."""
    
    # Results from the training (you can update these with actual results)
    cnn_results = {
        'accuracy': 75.00,  # 75%
        'precision': 75.00,
        'recall': 75.00,
        'f1_score': 75.00,
        'inference_time': 4.18,  # ms per image
        'training_time': 'Fast',  # Relative
        'model_size': 90  # MB (approximate)
    }
    
    vit_results = {
        'accuracy': 96.25,  # 96.25%
        'precision': 96.51,
        'recall': 96.25,
        'f1_score': 96.25,
        'inference_time': 5.31,  # ms per image
        'training_time': 'Slower',  # Relative
        'model_size': 327  # MB (approximate)
    }
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CNN vs ViT Performance Comparison\n(lr=0.001, batch_size=32, epochs=10)', 
                 fontsize=16, fontweight='bold')
    
    # 1. Accuracy Metrics Comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    cnn_values = [cnn_results['accuracy'], cnn_results['precision'], 
                  cnn_results['recall'], cnn_results['f1_score']]
    vit_values = [vit_results['accuracy'], vit_results['precision'], 
                  vit_results['recall'], vit_results['f1_score']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, cnn_values, width, label='CNN (ResNet50)', 
                    color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, vit_values, width, label='ViT', 
                    color='#e74c3c', alpha=0.8)
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score (%)')
    ax1.set_title('Performance Metrics Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    # 2. Inference Time Comparison
    models = ['CNN\n(ResNet50)', 'ViT']
    inference_times = [cnn_results['inference_time'], vit_results['inference_time']]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax2.bar(models, inference_times, color=colors, alpha=0.8)
    ax2.set_ylabel('Inference Time (ms per image)')
    ax2.set_title('Inference Speed Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, time in zip(bars, inference_times):
        ax2.annotate(f'{time:.2f} ms',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    # 3. Model Size Comparison
    model_sizes = [cnn_results['model_size'], vit_results['model_size']]
    bars = ax3.bar(models, model_sizes, color=colors, alpha=0.8)
    ax3.set_ylabel('Model Size (MB)')
    ax3.set_title('Model Size Comparison')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, size in zip(bars, model_sizes):
        ax3.annotate(f'{size} MB',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    # 4. Overall Performance Radar Chart
    categories = ['Accuracy', 'Speed\n(Inverse)', 'Model Size\n(Inverse)', 'Precision', 'Recall']
    
    # Normalize values for radar chart (0-100 scale)
    cnn_radar = [
        cnn_results['accuracy'],
        100 - (cnn_results['inference_time'] / max(cnn_results['inference_time'], vit_results['inference_time']) * 100),
        100 - (cnn_results['model_size'] / max(cnn_results['model_size'], vit_results['model_size']) * 100),
        cnn_results['precision'],
        cnn_results['recall']
    ]
    
    vit_radar = [
        vit_results['accuracy'],
        100 - (vit_results['inference_time'] / max(cnn_results['inference_time'], vit_results['inference_time']) * 100),
        100 - (vit_results['model_size'] / max(cnn_results['model_size'], vit_results['model_size']) * 100),
        vit_results['precision'],
        vit_results['recall']
    ]
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    cnn_radar += cnn_radar[:1]  # Complete the circle
    vit_radar += vit_radar[:1]
    angles += angles[:1]
    
    ax4.plot(angles, cnn_radar, 'o-', linewidth=2, label='CNN (ResNet50)', color='#3498db')
    ax4.fill(angles, cnn_radar, alpha=0.25, color='#3498db')
    ax4.plot(angles, vit_radar, 'o-', linewidth=2, label='ViT', color='#e74c3c')
    ax4.fill(angles, vit_radar, alpha=0.25, color='#e74c3c')
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories)
    ax4.set_ylim(0, 100)
    ax4.set_title('Overall Performance Comparison\n(Higher is Better)')
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('best_models/cnn_vit_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('cnn_vit_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("🎯 CNN vs ViT Comparison Summary")
    print("=" * 50)
    print(f"📊 Accuracy:")
    print(f"   CNN (ResNet50): {cnn_results['accuracy']:.2f}%")
    print(f"   ViT:            {vit_results['accuracy']:.2f}%")
    print(f"   Winner: {'ViT' if vit_results['accuracy'] > cnn_results['accuracy'] else 'CNN'} (+{abs(vit_results['accuracy'] - cnn_results['accuracy']):.2f}%)")
    
    print(f"\n⚡ Inference Speed:")
    print(f"   CNN (ResNet50): {cnn_results['inference_time']:.2f} ms/image")
    print(f"   ViT:            {vit_results['inference_time']:.2f} ms/image")
    print(f"   Winner: {'CNN' if cnn_results['inference_time'] < vit_results['inference_time'] else 'ViT'} ({abs(vit_results['inference_time'] - cnn_results['inference_time']):.2f} ms faster)")
    
    print(f"\n💾 Model Size:")
    print(f"   CNN (ResNet50): {cnn_results['model_size']} MB")
    print(f"   ViT:            {vit_results['model_size']} MB")
    print(f"   Winner: {'CNN' if cnn_results['model_size'] < vit_results['model_size'] else 'ViT'} ({abs(vit_results['model_size'] - cnn_results['model_size'])} MB smaller)")
    
    print(f"\n🏆 Overall Winner: ViT (significantly better accuracy despite larger size)")

if __name__ == "__main__":
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    create_cnn_vit_comparison()
    print("\n✅ Comparison chart saved as 'cnn_vit_comparison.png'") 