@echo off
echo 🐱 Quick Cat Classifier Test
echo ==========================

REM Disable wandb for quick testing
set WANDB_MODE=offline

REM Set default parameters for quick testing
set DATA_DIR=cat_dataset
set BATCH_SIZE=8
set EPOCHS=3
set LR=0.001
set OUTPUT_DIR=quick_test_output
set CAT_LIMIT=50
set NON_CAT_LIMIT=50

echo Running quick test with:
echo   Data directory: %DATA_DIR%
echo   Batch size: %BATCH_SIZE%
echo   Epochs: %EPOCHS%
echo   Learning rate: %LR%
echo   Output directory: %OUTPUT_DIR%
echo   Cat images: %CAT_LIMIT%
echo   Non-cat images: %NON_CAT_LIMIT%
echo   WANDB: DISABLED (offline mode)
echo.

REM Step 1: Download images
echo 📥 Step 1: Downloading images...
python download_images.py --output_dir %DATA_DIR% --cat_limit %CAT_LIMIT% --non_cat_limit %NON_CAT_LIMIT%

if %ERRORLEVEL% neq 0 (
    echo ❌ Error downloading images!
    pause
    exit /b 1
)

echo ✅ Images downloaded successfully!
echo.

REM Step 2: Run the training
echo 🚀 Step 2: Training models...
python cat_classifier.py --data-dir %DATA_DIR% --batch-size %BATCH_SIZE% --epochs %EPOCHS% --lr %LR% --output-dir %OUTPUT_DIR% --exp-name "quick-test"

if %ERRORLEVEL% neq 0 (
    echo ❌ Error during training!
    echo 🗑️ Cleaning up downloaded images...
    rmdir /s /q %DATA_DIR%
    pause
    exit /b 1
)

echo ✅ Training completed successfully!
echo.

REM Step 3: Clean up downloaded images to save space
echo 🗑️ Step 3: Cleaning up downloaded images to save space...
rmdir /s /q %DATA_DIR%
echo ✅ Downloaded images cleaned up!
echo.

echo ✅ Quick test completed!
echo 📁 Results saved to: %OUTPUT_DIR%
echo 🔍 To test inference, run: python inference.py
echo 💡 Note: Images were automatically cleaned up to save space
pause 