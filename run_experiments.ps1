# PowerShell script to run multiple experiments with different configurations
# This script compares CNN vs ViT performance with varying dataset sizes, epochs, and learning rates

# Load .env file and extract Wandb API key
if (Test-Path .env) {
    Get-Content .env | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            Set-Item -Path "Env:$key" -Value $value
        }
    }
    Write-Host "Loaded Wandb API key from .env file" -ForegroundColor Green
}
else {
    Write-Host "Warning: .env file not found. Please create it with your WANDB_API_KEY." -ForegroundColor Yellow
    exit
}

# Get API key from environment variable
$WANDB_API_KEY = $env:WANDB_API_KEY
if (-not $WANDB_API_KEY) {
    Write-Host "Error: WANDB_API_KEY not found in .env file" -ForegroundColor Red
    exit
}

# Storage optimization settings
$STORAGE_OPTIMIZATION = $true # Set to false to keep all datasets
$USE_SHARED_DATASET = $true # Set to true to use a single shared dataset for all experiments
$SHARED_DATASET_SIZE = 100 # Number of images per class for the shared dataset
$CLEANUP_AFTER_SIZE_CHANGE = $true # Clean up previous dataset when changing sizes
$CLEANUP_AFTER_ALL = $true # Clean up all datasets after all experiments

# Experiment configurations
$DATASET_SIZES = @(50, 100, 200, 500)  # Number of images per class
$EPOCHS = @(5, 10, 15)
$LEARNING_RATES = @(0.001, 0.0001)
$BATCH_SIZES = @(16, 32)

# Create directory for experiment outputs
$EXPERIMENTS_DIR = "experiments"
if (-not (Test-Path -Path $EXPERIMENTS_DIR)) {
    New-Item -ItemType Directory -Path $EXPERIMENTS_DIR
}

# Function to cleanup datasets
function Cleanup-Datasets {
    param (
        [string]$ExceptDataset = ""
    )
    
    Write-Host "Cleaning up datasets..." -ForegroundColor Yellow
    
    Get-ChildItem -Directory -Filter "cat_dataset_*" | ForEach-Object {
        if ($_.Name -ne $ExceptDataset) {
            Write-Host "Removing $($_.FullName)..." -ForegroundColor Yellow
            Remove-Item -Path $_.FullName -Recurse -Force
        }
    }
    
    Write-Host "Cleanup completed." -ForegroundColor Green
}

# Calculate total number of experiments for progress tracking
$total_experiments = 0
foreach ($dataset_size in $DATASET_SIZES) {
    foreach ($epochs in $EPOCHS) {
        foreach ($lr in $LEARNING_RATES) {
            foreach ($batch_size in $BATCH_SIZES) {
                # Skip some combinations to reduce total runtime
                if (($dataset_size -eq 500) -and ($epochs -eq 15) -and ($lr -eq 0.0001)) {
                    continue
                }
                $total_experiments++
            }
        }
    }
}

Write-Host "Starting $total_experiments experiments in total" -ForegroundColor Cyan
$current_experiment = 0

# Create a shared dataset if specified
if ($USE_SHARED_DATASET -and $STORAGE_OPTIMIZATION) {
    $shared_dataset_dir = "cat_dataset_shared"
    
    Write-Host ""
    Write-Host "========================================"
    Write-Host "Preparing shared dataset with $SHARED_DATASET_SIZE images per class" -ForegroundColor Cyan
    Write-Host "========================================"
    
    # Download dataset with specified size
    python download_images.py --cat_limit $SHARED_DATASET_SIZE --non_cat_limit $SHARED_DATASET_SIZE --output_dir $shared_dataset_dir
    
    Write-Host "Shared dataset ready at $shared_dataset_dir" -ForegroundColor Green
}

# Track current dataset size to avoid redundant downloads
$current_dataset_size = 0
$current_dataset_dir = ""

# Main experiment loop
foreach ($dataset_size in $DATASET_SIZES) {
    Write-Host ""
    Write-Host "========================================"
    Write-Host "Setting up experiments for dataset size: $dataset_size images per class" -ForegroundColor Cyan
    Write-Host "========================================"
    
    if ($USE_SHARED_DATASET -and $STORAGE_OPTIMIZATION) {
        # Use the shared dataset for all experiments
        $dataset_dir = $shared_dataset_dir
        Write-Host "Using shared dataset for experiments with dataset size $dataset_size" -ForegroundColor Green
    }
    else {
        $dataset_dir = "cat_dataset_$dataset_size"
        
        # Check if we need to download a new dataset
        if ($dataset_size -ne $current_dataset_size) {
            # Clean up previous dataset if needed
            if ($CLEANUP_AFTER_SIZE_CHANGE -and $STORAGE_OPTIMIZATION -and -not [string]::IsNullOrEmpty($current_dataset_dir)) {
                Write-Host "Cleaning up previous dataset: $current_dataset_dir" -ForegroundColor Yellow
                Remove-Item -Path $current_dataset_dir -Recurse -Force
            }
            
            # Download dataset with specified size
            Write-Host "Downloading new dataset with $dataset_size images per class" -ForegroundColor Cyan
            python download_images.py --cat_limit $dataset_size --non_cat_limit $dataset_size --output_dir $dataset_dir
            
            $current_dataset_size = $dataset_size
            $current_dataset_dir = $dataset_dir
        }
        else {
            Write-Host "Reusing existing dataset: $dataset_dir" -ForegroundColor Green
        }
    }
    
    foreach ($epochs in $EPOCHS) {
        foreach ($lr in $LEARNING_RATES) {
            foreach ($batch_size in $BATCH_SIZES) {
                # Skip some combinations to reduce total runtime
                if (($dataset_size -eq 500) -and ($epochs -eq 15) -and ($lr -eq 0.0001)) {
                    Write-Host "Skipping expensive combination: DS=$dataset_size, E=$epochs, LR=$lr, BS=$batch_size" -ForegroundColor Yellow
                    continue
                }
                
                # Update progress
                $current_experiment++
                $progress_percentage = [math]::Round(($current_experiment / $total_experiments) * 100, 1)
                
                # Create experiment name
                $exp_name = "cnn-vit-comparison-ds${dataset_size}-e${epochs}-lr${lr}-bs${batch_size}"
                $output_dir = "$EXPERIMENTS_DIR/$exp_name"
                
                # Progress bar
                $progress_bar = "[" + ("#" * [math]::Floor($progress_percentage / 2)) + (" " * [math]::Ceiling((100 - $progress_percentage) / 2)) + "]"
                
                Write-Host ""
                Write-Host "========================================"
                Write-Host "$progress_bar $progress_percentage% ($current_experiment/$total_experiments)" -ForegroundColor Green
                Write-Host "Running experiment: $exp_name" -ForegroundColor Cyan
                Write-Host "Dataset size: $dataset_size per class"
                Write-Host "Epochs: $epochs"
                Write-Host "Learning rate: $lr"
                Write-Host "Batch size: $batch_size"
                Write-Host "========================================"
                
                # Check for disk space before running the experiment
                $disk = Get-PSDrive -Name (Get-Location).Drive.Name
                $free_space_gb = [math]::Round($disk.Free / 1GB, 2)
                
                if ($free_space_gb -lt 5) {
                    Write-Host "Warning: Low disk space detected ($free_space_gb GB free). Cleaning up temporary files..." -ForegroundColor Red
                    
                    # Emergency cleanup of wandb runs
                    if (Test-Path "wandb") {
                        Get-ChildItem "wandb" -Directory | Where-Object { $_.Name -like "run-*" } | Select-Object -First 5 | ForEach-Object {
                            Write-Host "Removing old wandb run: $($_.Name)" -ForegroundColor Yellow
                            Remove-Item -Path $_.FullName -Recurse -Force
                        }
                    }
                }
                
                # Run experiment
                python cat_classifier.py `
                    --data-dir $dataset_dir `
                    --epochs $epochs `
                    --lr $lr `
                    --batch-size $batch_size `
                    --output-dir $output_dir `
                    --wandb-api-key $WANDB_API_KEY `
                    --wandb-project "cat-cnn-vit-comparison" `
                    --exp-name $exp_name `
                    --dataset-size $dataset_size
                
                Write-Host "Experiment completed: $exp_name" -ForegroundColor Green
                
                # Periodically check disk space
                $disk = Get-PSDrive -Name (Get-Location).Drive.Name
                $free_space_gb = [math]::Round($disk.Free / 1GB, 2)
                Write-Host "Remaining disk space: $free_space_gb GB" -ForegroundColor Cyan
            }
        }
    }
}

# Clean up all datasets after experiments if specified
if ($CLEANUP_AFTER_ALL -and $STORAGE_OPTIMIZATION) {
    Write-Host ""
    Write-Host "Cleaning up all datasets..." -ForegroundColor Yellow
    Cleanup-Datasets
}

Write-Host ""
Write-Host "All experiments completed!" -ForegroundColor Green
Write-Host "Check Weights & Biases dashboard for results and visualizations." 