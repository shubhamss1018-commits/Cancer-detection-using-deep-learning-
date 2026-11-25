"""
Example usage script for the Cancer Stage Prediction system.

This script demonstrates how to use the system programmatically
without command-line arguments.
"""

import torch
from pathlib import Path

from src.dataset import create_data_loaders
from src.model import create_model
from src.train import train_model
from src.evaluate import evaluate_model, print_evaluation_summary
from src.gradcam import visualize_gradcam
from src.utils import set_seed, plot_training_history

# Configuration
# Update this path to your dataset location
DATA_DIR = r"c:\Users\prava\Downloads\archive (1)\BreakHis - Breast Cancer Histopathological Database"
MAGNIFICATION = "400X"
BATCH_SIZE = 16  # Reduced for smaller GPUs
NUM_EPOCHS = 10  # Reduced for quick testing
IMG_SIZE = 224
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set random seed for reproducibility
set_seed(42)

print("=" * 70)
print("CANCER STAGE PREDICTION - EXAMPLE USAGE")
print("=" * 70)

# Step 1: Create data loaders
print("\n1. Loading dataset...")
train_loader, val_loader, test_loader, class_info = create_data_loaders(
    data_dir=DATA_DIR,
    magnification=MAGNIFICATION,
    batch_size=BATCH_SIZE,
    num_workers=2,  # Reduced for Windows
    train_split=0.7,
    val_split=0.15,
    img_size=IMG_SIZE,
    seed=42
)

print(f"   Classes: {class_info['num_classes']}")
print(f"   Stages: {class_info['num_stages']}")
print(f"   Risk Levels: {class_info['num_risk_levels']}")

# Step 2: Create model
print("\n2. Creating model...")
model = create_model(
    num_stages=class_info['num_stages'],
    num_risk_levels=class_info['num_risk_levels'],
    backbone='resnet50',
    pretrained=True,
    device=DEVICE
)

print(f"   Model created on {DEVICE}")
print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Step 3: Train model (optional - uncomment to train)
print("\n3. Training model...")
print("   (Skipping training in example - uncomment to train)")
"""
trained_model, history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=NUM_EPOCHS,
    learning_rate=1e-4,
    weight_decay=1e-4,
    device=DEVICE,
    save_dir='models',
    use_focal_loss=True
)

# Plot training history
plot_training_history(history, save_path='results/training_history.png')
"""

# Step 4: Evaluate model (requires trained model)
print("\n4. Evaluation...")
print("   (Requires trained model - load checkpoint first)")
"""
# Load trained model
checkpoint = torch.load('models/best_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate
results = evaluate_model(
    model=model,
    test_loader=test_loader,
    device=DEVICE,
    stage_names=['Stage 0', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4'],
    risk_names=['Low Risk', 'Medium Risk', 'High Risk']
)

print_evaluation_summary(results)
"""

# Step 5: Grad-CAM visualization (requires trained model and image)
print("\n5. Grad-CAM visualization...")
print("   (Requires trained model and image path)")
"""
# Example image path
image_path = "path/to/your/image.png"

visualize_gradcam(
    model=model,
    image_path=image_path,
    device=DEVICE,
    save_path='results/gradcam_example.png',
    task='stage',
    img_size=IMG_SIZE
)
"""

print("\n" + "=" * 70)
print("Example script completed!")
print("=" * 70)
print("\nTo use the full system, run:")
print("  python main.py --mode train --data_dir <path_to_dataset>")
print("  python main.py --mode evaluate --model_path models/best_model.pth --data_dir <path_to_dataset>")
print("  python main.py --mode gradcam --model_path models/best_model.pth --image_path <path_to_image> --data_dir <path_to_dataset>")
