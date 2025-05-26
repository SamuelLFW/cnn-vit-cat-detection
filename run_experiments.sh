#!/bin/bash
# Bash script to run multiple experiments with different configurations
# This script compares CNN vs ViT performance with varying dataset sizes, epochs, and learning rates

# Load .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo -e "\e[32mLoaded Wandb API key from .env file\e[0m"
else
    echo -e "\e[33mWarning: .env file not found. Please create it with your WANDB_API_KEY.\e[0m"
    exit 1
fi

# Check if API key exists
if [ -z "$WANDB_API_KEY" ]; then
    echo -e "\e[31mError: WANDB_API_KEY not found in .env file\e[0m"
    exit 1
fi

# Storage optimization settings
STORAGE_OPTIMIZATION=true # Set to false to keep all datasets
USE_SHARED_DATASET=true # Set to true to use a single shared dataset for all experiments
SHARED_DATASET_SIZE=100 # Number of images per class for the shared dataset
CLEANUP_AFTER_SIZE_CHANGE=true # Clean up previous dataset when changing sizes
CLEANUP_AFTER_ALL=true # Clean up all datasets after all experiments

# Experiment configurations
DATASET_SIZES=(50 100 200 500)  # Number of images per class
EPOCHS=(5 10 15)
LEARNING_RATES=(0.001 0.0001)
BATCH_SIZES=(16 32)

# Create directory for experiment outputs
EXPERIMENTS_DIR="experiments"
mkdir -p $EXPERIMENTS_DIR

# Function to cleanup datasets
cleanup_datasets() {
    except_dataset="$1"
    
    echo -e "\e[33mCleaning up datasets...\e[0m"
    
    for dir in cat_dataset_*; do
        if [ -d "$dir" ] && [ "$dir" != "$except_dataset" ]; then
            echo -e "\e[33mRemoving $dir...\e[0m"
            rm -rf "$dir"
        fi
    done
    
    echo -e "\e[32mCleanup completed.\e[0m"
}

# Function to check disk space and perform emergency cleanup if needed
check_disk_space() {
    # Get free disk space in GB (works on Linux and macOS)
    if [ "$(uname)" == "Darwin" ]; then
        # macOS
        free_space_gb=$(df -g . | tail -1 | awk '{print $4}')
    else
        # Linux
        free_space_gb=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    fi
    
    echo -e "\e[36mRemaining disk space: ${free_space_gb}GB\e[0m"
    
    # If less than 5GB free, perform emergency cleanup
    if (( $(echo "$free_space_gb < 5" | bc -l) )); then
        echo -e "\e[31mWarning: Low disk space detected (${free_space_gb}GB free). Cleaning up temporary files...\e[0m"
        
        # Emergency cleanup of wandb runs
        if [ -d "wandb" ]; then
            find wandb -type d -name "run-*" | head -5 | while read dir; do
                echo -e "\e[33mRemoving old wandb run: $dir\e[0m"
                rm -rf "$dir"
            done
        fi
    fi
}

# Calculate total experiments for progress tracking
total_experiments=0
for dataset_size in "${DATASET_SIZES[@]}"; do
    for epochs in "${EPOCHS[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
            for batch_size in "${BATCH_SIZES[@]}"; do
                # Skip some combinations to reduce total runtime
                if [ $dataset_size -eq 500 ] && [ $epochs -eq 15 ] && [ "$lr" = "0.0001" ]; then
                    continue
                fi
                total_experiments=$((total_experiments + 1))
            done
        done
    done
done

echo -e "\e[36mStarting $total_experiments experiments in total\e[0m"
current_experiment=0

# Create a shared dataset if specified
if [ "$USE_SHARED_DATASET" = true ] && [ "$STORAGE_OPTIMIZATION" = true ]; then
    shared_dataset_dir="cat_dataset_shared"
    
    echo ""
    echo "========================================"
    echo -e "\e[36mPreparing shared dataset with $SHARED_DATASET_SIZE images per class\e[0m"
    echo "========================================"
    
    # Download dataset with specified size
    python download_images.py --cat_limit $SHARED_DATASET_SIZE --non_cat_limit $SHARED_DATASET_SIZE --output_dir $shared_dataset_dir
    
    echo -e "\e[32mShared dataset ready at $shared_dataset_dir\e[0m"
fi

# Track current dataset size to avoid redundant downloads
current_dataset_size=0
current_dataset_dir=""

# Main experiment loop
for dataset_size in "${DATASET_SIZES[@]}"; do
    echo ""
    echo "========================================"
    echo -e "\e[36mSetting up experiments for dataset size: $dataset_size images per class\e[0m"
    echo "========================================"
    
    if [ "$USE_SHARED_DATASET" = true ] && [ "$STORAGE_OPTIMIZATION" = true ]; then
        # Use the shared dataset for all experiments
        dataset_dir=$shared_dataset_dir
        echo -e "\e[32mUsing shared dataset for experiments with dataset size $dataset_size\e[0m"
    else
        dataset_dir="cat_dataset_$dataset_size"
        
        # Check if we need to download a new dataset
        if [ "$dataset_size" -ne "$current_dataset_size" ]; then
            # Clean up previous dataset if needed
            if [ "$CLEANUP_AFTER_SIZE_CHANGE" = true ] && [ "$STORAGE_OPTIMIZATION" = true ] && [ ! -z "$current_dataset_dir" ]; then
                echo -e "\e[33mCleaning up previous dataset: $current_dataset_dir\e[0m"
                rm -rf "$current_dataset_dir"
            fi
            
            # Download dataset with specified size
            echo -e "\e[36mDownloading new dataset with $dataset_size images per class\e[0m"
            python download_images.py --cat_limit $dataset_size --non_cat_limit $dataset_size --output_dir $dataset_dir
            
            current_dataset_size=$dataset_size
            current_dataset_dir=$dataset_dir
        else
            echo -e "\e[32mReusing existing dataset: $dataset_dir\e[0m"
        fi
    fi
    
    for epochs in "${EPOCHS[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
            for batch_size in "${BATCH_SIZES[@]}"; do
                # Skip some combinations to reduce total runtime
                if [ $dataset_size -eq 500 ] && [ $epochs -eq 15 ] && [ "$lr" = "0.0001" ]; then
                    echo -e "\e[33mSkipping expensive combination: DS=$dataset_size, E=$epochs, LR=$lr, BS=$batch_size\e[0m"
                    continue
                fi
                
                # Update progress
                current_experiment=$((current_experiment + 1))
                progress_percentage=$(awk "BEGIN {printf \"%.1f\", ($current_experiment / $total_experiments) * 100}")
                
                # Create experiment name
                exp_name="cnn-vit-comparison-ds${dataset_size}-e${epochs}-lr${lr}-bs${batch_size}"
                output_dir="$EXPERIMENTS_DIR/$exp_name"
                
                # Create progress bar
                progress_bar="["
                filled_length=$(awk "BEGIN {printf \"%d\", $progress_percentage / 2}")
                empty_length=$(awk "BEGIN {printf \"%d\", 50 - $progress_percentage / 2}")
                
                for ((i=0; i<$filled_length; i++)); do
                    progress_bar+="#"
                done
                
                for ((i=0; i<$empty_length; i++)); do
                    progress_bar+=" "
                done
                
                progress_bar+="]"
                
                echo ""
                echo "========================================"
                echo -e "\e[32m$progress_bar $progress_percentage% ($current_experiment/$total_experiments)\e[0m"
                echo -e "\e[36mRunning experiment: $exp_name\e[0m"
                echo "Dataset size: $dataset_size per class"
                echo "Epochs: $epochs"
                echo "Learning rate: $lr"
                echo "Batch size: $batch_size"
                echo "========================================"
                
                # Check for disk space before running experiment
                check_disk_space
                
                # Run experiment
                python cat_classifier.py \
                    --data-dir $dataset_dir \
                    --epochs $epochs \
                    --lr $lr \
                    --batch-size $batch_size \
                    --output-dir $output_dir \
                    --wandb-api-key $WANDB_API_KEY \
                    --wandb-project "cat-classification-study" \
                    --exp-name $exp_name \
                    --dataset-size $dataset_size
                
                echo -e "\e[32mExperiment completed: $exp_name\e[0m"
                
                # Check disk space after experiment
                check_disk_space
            done
        done
    done
done

# Clean up all datasets after experiments if specified
if [ "$CLEANUP_AFTER_ALL" = true ] && [ "$STORAGE_OPTIMIZATION" = true ]; then
    echo ""
    echo -e "\e[33mCleaning up all datasets...\e[0m"
    cleanup_datasets
fi

echo ""
echo -e "\e[32mAll experiments completed!\e[0m"
echo "Check Weights & Biases dashboard for results and visualizations." 