"""
Complete setup verification script.
Checks all components are ready for training.
"""

import os
import sys
from pathlib import Path

print("=" * 70)
print("COMPLETE SETUP VERIFICATION")
print("=" * 70)

# Check 1: Python version
print("\n1. Checking Python version...")
python_version = sys.version_info
print(f"   Python {python_version.major}.{python_version.minor}.{python_version.micro}")
if python_version.major >= 3 and python_version.minor >= 8:
    print("   [OK] Python version is compatible")
else:
    print("   [WARNING] Python 3.8+ recommended")

# Check 2: Required packages
print("\n2. Checking required packages...")
required_packages = [
    'torch', 'torchvision', 'numpy', 'pandas', 
    'matplotlib', 'seaborn', 'sklearn', 'PIL', 'cv2', 'tqdm'
]
missing_packages = []
for package in required_packages:
    try:
        if package == 'PIL':
            __import__('PIL')
        elif package == 'cv2':
            __import__('cv2')
        elif package == 'sklearn':
            __import__('sklearn')
        else:
            __import__(package)
        print(f"   [OK] {package}")
    except ImportError:
        print(f"   [MISSING] {package}")
        missing_packages.append(package)

if missing_packages:
    print(f"\n   [ERROR] Missing packages: {', '.join(missing_packages)}")
    print("   Install with: pip install -r requirements.txt")
else:
    print("\n   [OK] All required packages are installed")

# Check 3: Dataset path
print("\n3. Checking dataset...")
dataset_root = r"c:\Users\prava\Downloads\archive (1)\BreakHis - Breast Cancer Histopathological Database"
expected_path = os.path.join(
    dataset_root,
    "dataset_cancer_v1",
    "dataset_cancer_v1",
    "classificacao_multiclasse",
    "400X"
)

if os.path.exists(expected_path):
    print(f"   [OK] Dataset found at: {expected_path}")
    classes = [d for d in os.listdir(expected_path) 
              if os.path.isdir(os.path.join(expected_path, d))]
    print(f"   [OK] Found {len(classes)} classes")
    
    # Count total images
    total_images = 0
    for cls in classes:
        cls_path = os.path.join(expected_path, cls)
        images = [f for f in os.listdir(cls_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        total_images += len(images)
    print(f"   [OK] Total images: {total_images}")
else:
    print(f"   [ERROR] Dataset not found at: {expected_path}")
    print("   Please verify the dataset path")

# Check 4: Project structure
print("\n4. Checking project structure...")
required_dirs = ['src', 'models', 'results', 'logs']
for dir_name in required_dirs:
    if os.path.exists(dir_name):
        print(f"   [OK] {dir_name}/ directory exists")
    else:
        print(f"   [WARNING] {dir_name}/ directory not found (will be created)")

# Check 5: Source files
print("\n5. Checking source files...")
required_files = [
    'src/dataset.py',
    'src/model.py',
    'src/train.py',
    'src/evaluate.py',
    'src/gradcam.py',
    'src/utils.py',
    'main.py'
]
missing_files = []
for file_path in required_files:
    if os.path.exists(file_path):
        print(f"   [OK] {file_path}")
    else:
        print(f"   [ERROR] {file_path} not found")
        missing_files.append(file_path)

if missing_files:
    print(f"\n   [ERROR] Missing files: {', '.join(missing_files)}")
else:
    print("\n   [OK] All source files present")

# Check 6: CUDA availability
print("\n6. Checking GPU availability...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   [OK] CUDA available")
        print(f"   [OK] GPU: {torch.cuda.get_device_name(0)}")
        print(f"   [OK] CUDA version: {torch.version.cuda}")
    else:
        print("   [INFO] CUDA not available, will use CPU")
        print("   [INFO] Training will be slower on CPU")
except:
    print("   [WARNING] Could not check CUDA availability")

# Check 7: Test dataset loading
print("\n7. Testing dataset loading...")
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from src.dataset import HistopathologyDataset, get_transforms
    
    transform = get_transforms('train', img_size=224)
    dataset = HistopathologyDataset(
        data_dir=dataset_root,
        magnification="400X",
        transform=transform
    )
    print(f"   [OK] Dataset loads successfully")
    print(f"   [OK] Total samples: {len(dataset)}")
    
    # Test sample
    if len(dataset) > 0:
        sample = dataset[0]
        image, targets = sample
        print(f"   [OK] Sample shape: {image.shape}")
        print(f"   [OK] Targets: Stage={targets['stage'].item()}, Risk={targets['progression_risk'].item()}")
except Exception as e:
    print(f"   [ERROR] Dataset loading failed: {e}")

# Summary
print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

if not missing_packages and not missing_files and os.path.exists(expected_path):
    print("\n[SUCCESS] Everything is ready! You can start training.")
    print("\nNext steps:")
    print("  1. Run: python test_dataset.py (to verify dataset)")
    print("  2. Run: python main.py --mode train --data_dir <path>")
    print("  3. Or use: train.bat (Windows batch script)")
else:
    print("\n[WARNING] Some issues found. Please fix them before training.")

print("=" * 70)
