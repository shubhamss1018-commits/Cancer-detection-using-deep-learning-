"""
Configuration file for the Cancer Stage Prediction system.
Update these paths and settings according to your setup.
"""

import os
from pathlib import Path

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# Path to the BreakHis dataset root directory
# Update this to match your dataset location
DATASET_ROOT = r"c:\Users\prava\Downloads\archive (1)\BreakHis - Breast Cancer Histopathological Database"

# Magnification level to use (40X, 100X, 200X, 400X)
MAGNIFICATION = "400X"

# Image preprocessing
IMG_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
NORMALIZE_STD = [0.229, 0.224, 0.225]   # ImageNet std

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Data splitting
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15  # Automatically calculated as 1 - TRAIN_SPLIT - VAL_SPLIT

# Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4  # Use 0 on Windows if you encounter issues

# Model architecture
BACKBONE = 'resnet50'  # Options: 'resnet50', 'efficientnet'
PRETRAINED = True
USE_FOCAL_LOSS = True

# Loss weights
STAGE_LOSS_WEIGHT = 1.0
PROGRESSION_LOSS_WEIGHT = 0.5

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Number of classes
NUM_STAGES = 5  # Cancer stages 0-4
NUM_RISK_LEVELS = 3  # Low, Medium, High
FEATURE_DIM = 512
DROPOUT = 0.5

# ============================================================================
# PATHS AND DIRECTORIES
# ============================================================================

# Project root
PROJECT_ROOT = Path(__file__).parent

# Output directories
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"

# Create directories if they don't exist
for dir_path in [MODELS_DIR, RESULTS_DIR, LOGS_DIR, DATA_DIR]:
    dir_path.mkdir(exist_ok=True)

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

RANDOM_SEED = 42

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

# Class names for visualization
STAGE_NAMES = ['Stage 0', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']
RISK_NAMES = ['Low Risk', 'Medium Risk', 'High Risk']

# Original BreakHis class names
BREAKHIS_CLASSES = [
    'adenosis',
    'ductal_carcinoma',
    'fibroadenoma',
    'lobular_carcinoma',
    'mucinous_carcinoma',
    'papillary_carcinoma',
    'phyllodes_tumor',
    'tubular_adenoma'
]

# Cancer stage mapping
BREAKHIS_TO_STAGE = {
    'adenosis': 0,              # Benign
    'fibroadenoma': 0,          # Benign
    'tubular_adenoma': 0,       # Benign
    'phyllodes_tumor': 1,       # Borderline/Early
    'papillary_carcinoma': 1,   # Early stage
    'mucinous_carcinoma': 2,    # Early-Mid stage
    'lobular_carcinoma': 3,    # Mid-Advanced stage
    'ductal_carcinoma': 4,     # Advanced stage
}

# ============================================================================
# GRAD-CAM CONFIGURATION
# ============================================================================

GRADCAM_ALPHA = 0.4  # Transparency for heatmap overlay
GRADCAM_COLORMAP = 'jet'  # Colormap for heatmap

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_dataset_path():
    """Get the full path to the dataset."""
    return os.path.join(
        DATASET_ROOT,
        "dataset_cancer_v1",
        "dataset_cancer_v1",
        "classificacao_multiclasse",
        MAGNIFICATION
    )

def verify_dataset():
    """Verify that the dataset exists and is accessible."""
    dataset_path = get_dataset_path()
    if os.path.exists(dataset_path):
        return True, dataset_path
    return False, dataset_path
