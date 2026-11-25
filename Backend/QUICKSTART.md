# Quick Start Guide

## ✅ Dataset Verification

The dataset has been verified and is ready to use!

**Dataset Location:**
```
c:\Users\prava\Downloads\archive (1)\BreakHis - Breast Cancer Histopathological Database\dataset_cancer_v1\dataset_cancer_v1\classificacao_multiclasse
```

**Dataset Statistics (400X magnification):**
- Total samples: **1,820 images**
- Classes: **8** (adenosis, ductal_carcinoma, fibroadenoma, lobular_carcinoma, mucinous_carcinoma, papillary_carcinoma, phyllodes_tumor, tubular_adenoma)
- Train/Val/Test split: **70% / 15% / 15%**
- Available magnifications: **40X, 100X, 200X, 400X**

## 🚀 Getting Started

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Verify Dataset Loading

Test that the dataset loads correctly:

```bash
python test_dataset.py
```

Expected output:
- [OK] Dataset path is correct!
- [OK] Dataset created successfully!
- [OK] Data loaders created successfully!

### Step 3: Train the Model

Start training with default settings:

```bash
python main.py --mode train ^
    --data_dir "c:\Users\prava\Downloads\archive (1)\BreakHis - Breast Cancer Histopathological Database" ^
    --magnification 400X ^
    --batch_size 32 ^
    --num_epochs 50 ^
    --learning_rate 1e-4
```

**For smaller GPUs or testing:**
```bash
python main.py --mode train ^
    --data_dir "c:\Users\prava\Downloads\archive (1)\BreakHis - Breast Cancer Histopathological Database" ^
    --magnification 400X ^
    --batch_size 16 ^
    --num_epochs 10 ^
    --num_workers 0
```

### Step 4: Evaluate the Model

After training, evaluate on the test set:

```bash
python main.py --mode evaluate ^
    --data_dir "c:\Users\prava\Downloads\archive (1)\BreakHis - Breast Cancer Histopathological Database" ^
    --model_path "models/best_model.pth" ^
    --magnification 400X
```

### Step 5: Generate Grad-CAM Visualizations

Create explainable AI visualizations:

```bash
python main.py --mode gradcam ^
    --data_dir "c:\Users\prava\Downloads\archive (1)\BreakHis - Breast Cancer Histopathological Database" ^
    --model_path "models/best_model.pth" ^
    --image_path "path/to/your/image.png" ^
    --magnification 400X
```

## 📊 Expected Results

After training, you'll find:

1. **Model Checkpoints** in `models/`:
   - `best_model.pth` - Best model based on validation accuracy
   - `checkpoint_epoch_*.pth` - Epoch checkpoints

2. **Results** in `results/run_YYYYMMDD_HHMMSS/`:
   - `training_history.png` - Training curves
   - `sample_predictions.png` - Sample predictions
   - `evaluation/` - Confusion matrices and reports
   - `gradcam/` - Grad-CAM visualizations

## ⚙️ Configuration Options

### Magnification Levels
- `--magnification 40X` - Lower resolution, faster training
- `--magnification 100X` - Medium resolution
- `--magnification 200X` - Higher resolution
- `--magnification 400X` - Highest resolution (recommended)

### Model Architectures
- `--backbone resnet50` - ResNet50 (default, recommended)
- `--backbone efficientnet` - EfficientNet (if available)

### Training Parameters
- `--batch_size 32` - Batch size (reduce if out of memory)
- `--num_epochs 50` - Number of training epochs
- `--learning_rate 1e-4` - Learning rate
- `--num_workers 4` - Data loader workers (use 0 on Windows if issues)

## 🔧 Troubleshooting

### Issue: Out of Memory
**Solution:** Reduce batch size
```bash
--batch_size 16  # or 8
```

### Issue: Slow Data Loading
**Solution:** Reduce workers or use 0 on Windows
```bash
--num_workers 0
```

### Issue: Dataset Not Found
**Solution:** Verify the path and use absolute path
```bash
--data_dir "c:\Users\prava\Downloads\archive (1)\BreakHis - Breast Cancer Histopathological Database"
```

### Issue: CUDA Out of Memory
**Solution:** Use CPU or reduce batch size
```bash
--cpu  # Force CPU usage
--batch_size 8
```

## 📝 Notes

- Training time: ~2-4 hours on GPU for 50 epochs (depends on hardware)
- The model automatically handles class imbalance using Focal Loss
- All results are saved with timestamps for easy tracking
- The system supports resuming training from checkpoints

## 🎯 Next Steps

1. **Train the model** with your preferred settings
2. **Evaluate** on the test set to see performance metrics
3. **Generate Grad-CAM** visualizations for interpretability
4. **Analyze results** in the results directory
5. **Fine-tune** hyperparameters if needed

---

**Ready to start?** Run the training command above and monitor the progress!
