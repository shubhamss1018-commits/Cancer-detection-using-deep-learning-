# AI Tool for Automatic Cancer Stage and Progression Prediction

An end-to-end deep learning system for automatic cancer stage classification and progression risk estimation using histopathology images. This tool provides interpretable predictions with Grad-CAM visualizations for clinical decision support.

## 🎯 Project Overview

This system uses transfer learning with CNN architectures (ResNet50/EfficientNet) to:
1. **Classify cancer stages** (0-4) from histopathology image patches
2. **Estimate progression risk** (Low/Medium/High)
3. **Generate explainable visualizations** using Grad-CAM to highlight important tissue regions
4. **Produce research-ready outputs** including metrics, confusion matrices, and visualizations

## 📋 Features

- **Multi-task Learning**: Simultaneous prediction of cancer stage and progression risk
- **Transfer Learning**: Pretrained models (ResNet50/EfficientNet) for robust feature extraction
- **Class Imbalance Handling**: Focal loss for addressing imbalanced datasets
- **Explainable AI**: Grad-CAM visualizations for model interpretability
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and classification reports
- **Research-Ready**: Publication-quality visualizations and metrics

## 🏗️ Project Structure

```
.
├── src/
│   ├── dataset.py          # Data loading and preprocessing
│   ├── model.py            # Model architecture
│   ├── train.py            # Training loop and loss functions
│   ├── evaluate.py         # Evaluation metrics and visualization
│   ├── gradcam.py          # Grad-CAM implementation
│   └── utils.py            # Utility functions
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── data/                   # Dataset directory (to be created)
├── models/                 # Saved model checkpoints
├── results/                # Training results and visualizations
└── logs/                   # Training logs
```

## 🚀 Installation

1. **Clone or download this repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Prepare your dataset**:
   - The dataset should be located at:
     ```
     c:\Users\prava\Downloads\archive (1)\BreakHis - Breast Cancer Histopathological Database\dataset_cancer_v1\dataset_cancer_v1\classificacao_multiclasse\
     ```
   - Or update the path in `config.py` or use `--data_dir` argument
   - Expected structure:
     ```
     dataset_cancer_v1/
       dataset_cancer_v1/
         classificacao_multiclasse/
           400X/
             adenosis/
             ductal_carcinoma/
             fibroadenoma/
             ...
     ```

## 📖 Usage

### 1. Training

Train a new model from scratch:

```bash
python main.py --mode train \
    --data_dir "path/to/your/dataset" \
    --magnification 400X \
    --batch_size 32 \
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --backbone resnet50
```

**Arguments:**
- `--data_dir`: Path to your dataset directory
- `--magnification`: Magnification level (40X, 100X, 200X, 400X)
- `--batch_size`: Batch size for training
- `--num_epochs`: Number of training epochs
- `--learning_rate`: Learning rate
- `--backbone`: Model architecture (resnet50 or efficientnet)
- `--img_size`: Input image size (default: 224)
- `--use_focal_loss`: Use focal loss for class imbalance (default: True)

### 2. Evaluation

Evaluate a trained model on the test set:

```bash
python main.py --mode evaluate \
    --data_dir "path/to/your/dataset" \
    --model_path "models/best_model.pth" \
    --magnification 400X
```

This will generate:
- Confusion matrices for stage and progression risk
- Classification reports
- Sample prediction visualizations
- Detailed metrics summary

### 3. Grad-CAM Visualization

Generate explainable visualizations for individual images:

```bash
python main.py --mode gradcam \
    --data_dir "path/to/your/dataset" \
    --model_path "models/best_model.pth" \
    --image_path "path/to/image.png" \
    --magnification 400X
```

For batch processing:

```bash
python main.py --mode gradcam \
    --data_dir "path/to/your/dataset" \
    --model_path "models/best_model.pth" \
    --image_path "path/to/image/directory" \
    --num_samples 10
```

## 📊 Dataset

This project is designed to work with the **BreakHis** (Breast Cancer Histopathological Database) dataset, which contains:
- **8 classes**: adenosis, ductal_carcinoma, fibroadenoma, lobular_carcinoma, mucinous_carcinoma, papillary_carcinoma, phyllodes_tumor, tubular_adenoma
- **4 magnifications**: 40X, 100X, 200X, 400X
- **Multi-class classification** for cancer staging

### Cancer Stage Mapping

The system maps BreakHis classes to cancer stages:
- **Stage 0** (Benign): adenosis, fibroadenoma, tubular_adenoma
- **Stage 1** (Early): phyllodes_tumor, papillary_carcinoma
- **Stage 2** (Early-Mid): mucinous_carcinoma
- **Stage 3** (Mid-Advanced): lobular_carcinoma
- **Stage 4** (Advanced): ductal_carcinoma

### Progression Risk Mapping

- **Low Risk**: Stages 0-1
- **Medium Risk**: Stage 2
- **High Risk**: Stages 3-4

## 🧠 Model Architecture

The model consists of:
1. **Backbone**: Pretrained ResNet50 or EfficientNet for feature extraction
2. **Feature Extractor**: Fully connected layers to project features
3. **Stage Classifier**: Multi-class head for cancer stage prediction (5 classes)
4. **Progression Classifier**: Multi-class head for risk estimation (3 classes)

## 📈 Training Details

- **Loss Function**: Multi-task loss combining stage and progression risk predictions
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: Adaptive reduction on plateau
- **Data Augmentation**: Random crops, flips, rotations, color jitter, random erasing
- **Class Imbalance**: Focal loss with gamma=2.0

## 📊 Outputs

The system generates:

1. **Model Checkpoints**: Best model and epoch checkpoints
2. **Training Curves**: Loss and accuracy plots
3. **Evaluation Metrics**:
   - Accuracy, Precision, Recall, F1-Score
   - Confusion matrices
   - Classification reports
4. **Visualizations**:
   - Sample predictions with ground truth
   - Grad-CAM heatmaps
   - Overlaid visualizations
5. **Research Reports**: Text files with detailed metrics

## 🔬 Research Context

This tool is designed for:
- **Medical Imaging Research**: Histopathology analysis and cancer staging
- **Clinical Decision Support**: Assisting pathologists with automated analysis
- **Explainable AI**: Providing interpretable predictions for healthcare applications
- **Academic Research**: Publication-ready metrics and visualizations

## ⚙️ Configuration

Key parameters can be adjusted in `main.py`:
- Image size and augmentation strategies
- Model architecture and pretrained weights
- Loss function weights
- Learning rate and training schedule
- Evaluation metrics

## 📝 Citation

If you use this code in your research, please cite:

```
AI Tool for Automatic Cancer Stage and Progression Prediction using Histopathology Images
Deep Learning-based Decision Support System with Explainable AI
```

## 🤝 Contributing

This is a research project. For improvements or issues, please create an issue or pull request.

## 📄 License

This project is intended for research and educational purposes.

## 🙏 Acknowledgments

- BreakHis dataset for histopathology images
- PyTorch and torchvision for deep learning framework
- Medical imaging research community

## 📧 Contact

For questions or support, please refer to the project documentation or create an issue.

---

**Note**: This tool is for research purposes and should not be used as the sole basis for clinical decisions. Always consult with qualified medical professionals for diagnosis and treatment decisions.
