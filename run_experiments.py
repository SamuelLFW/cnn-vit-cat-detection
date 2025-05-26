#!/usr/bin/env python3
"""
Run multiple experiments with different settings for CNN and ViT models.
This script will create separate wandb runs for each model and setting combination.
"""

import subprocess
import sys
import time
from itertools import product

def run_experiment(model, batch_size, epochs, lr, data_dir="cat_dataset", wandb_project="cat-classification-experiments"):
    """Run a single experiment with the given parameters."""
    
    exp_name = f"{model}-bs{batch_size}-e{epochs}-lr{lr}"
    output_dir = f"experiments/{exp_name}"
    
    cmd = [
        sys.executable, "cat_classifier.py",
        "--model", model,
        "--batch-size", str(batch_size),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--data-dir", data_dir,
        "--output-dir", output_dir,
        "--wandb-project", wandb_project,
        "--exp-name", exp_name
    ]
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {exp_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ Experiment {exp_name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment {exp_name} failed with error: {e}")
        return False

def main():
    """Run all experiments."""
    
    # Experiment settings
    models = ['cnn', 'vit']
    batch_sizes = [8, 16, 32]
    epochs_list = [5, 10, 15]
    learning_rates = [0.0001, 0.001, 0.01]
    
    # Data directory
    data_dir = "cat_dataset"
    wandb_project = "cat-classification-experiments"
    
    print("🚀 Starting CNN vs ViT Experiments")
    print(f"Models: {models}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Epochs: {epochs_list}")
    print(f"Learning rates: {learning_rates}")
    print(f"Total experiments: {len(models) * len(batch_sizes) * len(epochs_list) * len(learning_rates)}")
    
    # Track results
    successful_experiments = []
    failed_experiments = []
    
    # Run all combinations
    for model, batch_size, epochs, lr in product(models, batch_sizes, epochs_list, learning_rates):
        success = run_experiment(
            model=model,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            data_dir=data_dir,
            wandb_project=wandb_project
        )
        
        if success:
            successful_experiments.append(f"{model}-bs{batch_size}-e{epochs}-lr{lr}")
        else:
            failed_experiments.append(f"{model}-bs{batch_size}-e{epochs}-lr{lr}")
        
        # Small delay between experiments
        time.sleep(2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful experiments: {len(successful_experiments)}")
    print(f"❌ Failed experiments: {len(failed_experiments)}")
    
    if successful_experiments:
        print(f"\n✅ Successful:")
        for exp in successful_experiments:
            print(f"  - {exp}")
    
    if failed_experiments:
        print(f"\n❌ Failed:")
        for exp in failed_experiments:
            print(f"  - {exp}")
    
    print(f"\n🎯 Check your results on wandb: https://wandb.ai/your-username/{wandb_project}")

if __name__ == "__main__":
    main() 