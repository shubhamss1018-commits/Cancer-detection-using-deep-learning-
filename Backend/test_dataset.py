"""
Test script to verify dataset loading works correctly.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import HistopathologyDataset, create_data_loaders, get_transforms

# Dataset path
DATA_DIR = r"c:\Users\prava\Downloads\archive (1)\BreakHis - Breast Cancer Histopathological Database"
MAGNIFICATION = "400X"

print("=" * 70)
print("DATASET VERIFICATION TEST")
print("=" * 70)

# Test 1: Check if dataset path exists
print(f"\n1. Checking dataset path...")
expected_path = os.path.join(
    DATA_DIR,
    "dataset_cancer_v1",
    "dataset_cancer_v1",
    "classificacao_multiclasse",
    MAGNIFICATION
)
print(f"   Expected path: {expected_path}")
print(f"   Path exists: {os.path.exists(expected_path)}")

if os.path.exists(expected_path):
    print("   [OK] Dataset path is correct!")
    
    # List classes
    classes = [d for d in os.listdir(expected_path) 
              if os.path.isdir(os.path.join(expected_path, d))]
    print(f"   Found {len(classes)} classes: {sorted(classes)}")
    
    # Count images in first class
    if classes:
        first_class = classes[0]
        class_path = os.path.join(expected_path, first_class)
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        print(f"   Sample: {first_class} has {len(images)} images")
        if images:
            print(f"   First image: {images[0]}")
else:
    print("   [ERROR] Dataset path not found!")
    print("   Please check the path and update DATA_DIR if needed.")

# Test 2: Try to create dataset
print(f"\n2. Testing dataset creation...")
try:
    transform = get_transforms('train', img_size=224)
    dataset = HistopathologyDataset(
        data_dir=DATA_DIR,
        magnification=MAGNIFICATION,
        transform=transform
    )
    print(f"   [OK] Dataset created successfully!")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Classes: {len(dataset.class_to_idx)}")
    print(f"   Class mapping: {dataset.class_to_idx}")
    
    # Test getting a sample
    if len(dataset) > 0:
        sample = dataset[0]
        image, targets = sample
        print(f"   [OK] Sample loaded successfully!")
        print(f"   Image shape: {image.shape}")
        print(f"   Stage: {targets['stage'].item()}")
        print(f"   Progression Risk: {targets['progression_risk'].item()}")
        print(f"   Class: {targets['class'].item()}")
        
except Exception as e:
    print(f"   [ERROR] Error creating dataset: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Try to create data loaders
print(f"\n3. Testing data loader creation...")
try:
    train_loader, val_loader, test_loader, class_info = create_data_loaders(
        data_dir=DATA_DIR,
        magnification=MAGNIFICATION,
        batch_size=4,  # Small batch for testing
        num_workers=0,  # Use 0 for Windows compatibility
        train_split=0.7,
        val_split=0.15,
        img_size=224,
        seed=42
    )
    print(f"   [OK] Data loaders created successfully!")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Test loading a batch
    images, targets = next(iter(train_loader))
    print(f"   [OK] Batch loaded successfully!")
    print(f"   Batch size: {images.shape[0]}")
    print(f"   Image shape: {images.shape}")
    print(f"   Stage targets: {targets['stage']}")
    print(f"   Progression targets: {targets['progression_risk']}")
    
except Exception as e:
    print(f"   [ERROR] Error creating data loaders: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
