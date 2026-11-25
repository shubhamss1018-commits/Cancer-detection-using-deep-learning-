@echo off
REM Training script for Cancer Stage Prediction

echo ========================================
echo Training Cancer Stage Prediction Model
echo ========================================
echo.

python main.py --mode train ^
    --data_dir "c:\Users\prava\Downloads\archive (1)\BreakHis - Breast Cancer Histopathological Database" ^
    --magnification 400X ^
    --batch_size 32 ^
    --num_epochs 50 ^
    --learning_rate 1e-4 ^
    --backbone resnet50 ^
    --num_workers 0

pause
