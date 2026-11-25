@echo off
REM Evaluation script for Cancer Stage Prediction

echo ========================================
echo Evaluating Cancer Stage Prediction Model
echo ========================================
echo.

if not exist "models\best_model.pth" (
    echo ERROR: Model file not found!
    echo Please train the model first using train.bat
    pause
    exit /b 1
)

python main.py --mode evaluate ^
    --data_dir "c:\Users\prava\Downloads\archive (1)\BreakHis - Breast Cancer Histopathological Database" ^
    --model_path "models\best_model.pth" ^
    --magnification 400X ^
    --batch_size 32 ^
    --num_workers 0

pause
