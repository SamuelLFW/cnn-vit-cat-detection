@echo off
echo 🧪 Quick CNN vs ViT Experiments
echo ================================

REM Set common parameters
set DATA_DIR=cat_dataset
set WANDB_PROJECT=cat-classification-experiments
set EPOCHS=5

echo Running quick experiments with:
echo   Data directory: %DATA_DIR%
echo   Epochs: %EPOCHS%
echo   Wandb project: %WANDB_PROJECT%
echo.

REM Create experiments directory
if not exist experiments mkdir experiments

echo 📥 Step 1: Downloading dataset (if needed)...
python download_images.py --output_dir %DATA_DIR% --cat_limit 100 --non_cat_limit 100

if %ERRORLEVEL% neq 0 (
    echo ❌ Error downloading dataset!
    pause
    exit /b 1
)

echo ✅ Dataset ready!
echo.

REM Experiment 1: CNN with batch size 16, lr 0.001
echo 🔬 Experiment 1: CNN (bs=16, lr=0.001)
python cat_classifier.py --model cnn --batch-size 16 --epochs %EPOCHS% --lr 0.001 --data-dir %DATA_DIR% --output-dir experiments/cnn-bs16-e%EPOCHS%-lr0.001 --wandb-project %WANDB_PROJECT% --exp-name cnn-bs16-e%EPOCHS%-lr0.001

REM Experiment 2: ViT with batch size 16, lr 0.0001 (lower LR for ViT)
echo 🔬 Experiment 2: ViT (bs=16, lr=0.0001)
python cat_classifier.py --model vit --batch-size 16 --epochs %EPOCHS% --lr 0.001 --data-dir %DATA_DIR% --output-dir experiments/vit-bs16-e%EPOCHS%-lr0.001 --wandb-project %WANDB_PROJECT% --exp-name vit-bs16-e%EPOCHS%-lr0.001

REM Experiment 3: CNN with batch size 32, lr 0.001
echo 🔬 Experiment 3: CNN (bs=32, lr=0.001)
python cat_classifier.py --model cnn --batch-size 32 --epochs %EPOCHS% --lr 0.001 --data-dir %DATA_DIR% --output-dir experiments/cnn-bs32-e%EPOCHS%-lr0.001 --wandb-project %WANDB_PROJECT% --exp-name cnn-bs32-e%EPOCHS%-lr0.001

REM Experiment 4: ViT with batch size 32, lr 0.0001
echo 🔬 Experiment 4: ViT (bs=32, lr=0.0001)
python cat_classifier.py --model vit --batch-size 32 --epochs %EPOCHS% --lr 0.001 --data-dir %DATA_DIR% --output-dir experiments/vit-bs32-e%EPOCHS%-lr0.001 --wandb-project %WANDB_PROJECT% --exp-name vit-bs32-e%EPOCHS%-lr0.001

REM Experiment 5: CNN with higher learning rate
echo 🔬 Experiment 5: CNN (bs=16, lr=0.01)
python cat_classifier.py --model cnn --batch-size 16 --epochs %EPOCHS% --lr 0.01 --data-dir %DATA_DIR% --output-dir experiments/cnn-bs16-e%EPOCHS%-lr0.01 --wandb-project %WANDB_PROJECT% --exp-name cnn-bs16-e%EPOCHS%-lr0.01

REM Experiment 6: ViT with higher learning rate
echo 🔬 Experiment 6: ViT (bs=16, lr=0.01)
python cat_classifier.py --model vit --batch-size 16 --epochs %EPOCHS% --lr 0.01 --data-dir %DATA_DIR% --output-dir experiments/vit-bs16-e%EPOCHS%-lr0.01 --wandb-project %WANDB_PROJECT% --exp-name vit-bs16-e%EPOCHS%-lr0.01

echo.
echo ✅ All experiments completed!
echo 📊 Check your results on wandb: https://wandb.ai/your-username/%WANDB_PROJECT%
echo 📁 Local results saved in: experiments/
echo.
echo 🗑️ Cleaning up dataset to save space...
rmdir /s /q %DATA_DIR%
echo ✅ Dataset cleaned up!

pause 