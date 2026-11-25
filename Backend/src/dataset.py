"""
Dataset loading and preprocessing module for histopathology images.
Handles data loading, augmentation, and preparation for training.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, Optional, Dict, List


class HistopathologyDataset(Dataset):
    """
    Custom dataset class for histopathology images.
    Supports multi-class cancer stage classification.
    """
    
    def __init__(
        self,
        data_dir: str,
        magnification: str = "400X",
        transform: Optional[transforms.Compose] = None,
        class_to_idx: Optional[Dict[str, int]] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Root directory containing the dataset
            magnification: Magnification level (40X, 100X, 200X, 400X)
            transform: Optional transform to apply to images
            class_to_idx: Optional mapping from class names to indices
        """
        self.data_dir = data_dir
        self.magnification = magnification
        self.transform = transform
        
        # Resolve the path to the specific magnification folder.
        # Support both of these common layouts:
        #   1) <data_dir>/dataset_cancer_v1/dataset_cancer_v1/classificacao_multiclasse/400X
        #   2) <data_dir>/dataset_cancer_v1/classificacao_multiclasse/400X
        candidate_roots = [
            os.path.join(data_dir, "dataset_cancer_v1", "dataset_cancer_v1"),
            os.path.join(data_dir, "dataset_cancer_v1"),
        ]
        mag_path = None
        for root in candidate_roots:
            candidate = os.path.join(root, "classificacao_multiclasse", magnification)
            if os.path.exists(candidate):
                mag_path = candidate
                break
        
        # Fallback to the original double-nested structure if nothing matched;
        # this keeps behavior consistent for the original layout.
        if mag_path is None:
            mag_path = os.path.join(
                data_dir,
                "dataset_cancer_v1",
                "dataset_cancer_v1",
                "classificacao_multiclasse",
                magnification
            )
        
        self.mag_path = mag_path
        
        # Define class mapping to cancer stages
        # Mapping BreakHis classes to cancer stages (0-4)
        # Stage 0: Benign/Pre-cancerous
        # Stage 1-2: Early stage cancer
        # Stage 3-4: Advanced stage cancer
        self.breakhis_to_stage = {
            'adenosis': 0,              # Benign
            'fibroadenoma': 0,          # Benign
            'tubular_adenoma': 0,       # Benign
            'phyllodes_tumor': 1,       # Borderline/Early
            'papillary_carcinoma': 1,   # Early stage
            'mucinous_carcinoma': 2,    # Early-Mid stage
            'lobular_carcinoma': 3,     # Mid-Advanced stage
            'ductal_carcinoma': 4,      # Advanced stage
        }
        
        # If class_to_idx not provided, create from available classes
        if class_to_idx is None:
            if os.path.exists(self.mag_path):
                classes = sorted([d for d in os.listdir(self.mag_path) 
                                if os.path.isdir(os.path.join(self.mag_path, d))])
                self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
            else:
                # Fallback to default classes
                self.class_to_idx = {
                    'adenosis': 0, 'ductal_carcinoma': 1, 'fibroadenoma': 2,
                    'lobular_carcinoma': 3, 'mucinous_carcinoma': 4,
                    'papillary_carcinoma': 5, 'phyllodes_tumor': 6,
                    'tubular_adenoma': 7
                }
        else:
            self.class_to_idx = class_to_idx
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Load all image paths and labels
        self.samples = []
        self._load_samples()
        
    def _load_samples(self):
        """Load all image paths and corresponding labels."""
        if not os.path.exists(self.mag_path):
            raise ValueError(f"Dataset path does not exist: {self.mag_path}")
        
        for class_name in self.class_to_idx.keys():
            class_dir = os.path.join(self.mag_path, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            class_idx = self.class_to_idx[class_name]
            # Get cancer stage from mapping
            cancer_stage = self.breakhis_to_stage.get(class_name, 0)
            
            # Load all images in this class
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append({
                        'path': img_path,
                        'class_idx': class_idx,
                        'class_name': class_name,
                        'cancer_stage': cancer_stage
                    })
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get a sample from the dataset.
        
        Returns:
            image: Preprocessed image tensor
            targets: Dictionary containing:
                - 'stage': Cancer stage (0-4)
                - 'class': Original class index
                - 'progression_risk': Risk level (0=low, 1=medium, 2=high)
        """
        sample = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(sample['path']).convert('RGB')
        except Exception as e:
            print(f"Error loading image {sample['path']}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get targets
        stage = sample['cancer_stage']
        class_idx = sample['class_idx']
        
        # Calculate progression risk based on stage
        # Low: 0-1, Medium: 2, High: 3-4
        if stage <= 1:
            progression_risk = 0  # Low
        elif stage == 2:
            progression_risk = 1  # Medium
        else:
            progression_risk = 2  # High
        
        targets = {
            'stage': torch.tensor(stage, dtype=torch.long),
            'class': torch.tensor(class_idx, dtype=torch.long),
            'progression_risk': torch.tensor(progression_risk, dtype=torch.long)
        }
        
        return image, targets


def get_transforms(
    phase: str = 'train',
    img_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> transforms.Compose:
    """
    Get data transforms for training/validation/test.
    
    Args:
        phase: 'train', 'val', or 'test'
        img_size: Target image size
        mean: Normalization mean for ImageNet pretrained models
        std: Normalization std for ImageNet pretrained models
    
    Returns:
        Compose transform object
    """
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.33))
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


def create_data_loaders(
    data_dir: str,
    magnification: str = "400X",
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.7,
    val_split: float = 0.15,
    img_size: int = 224,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_dir: Root directory of the dataset
        magnification: Magnification level to use
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        img_size: Target image size
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_info)
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create full dataset to get class mappings
    full_dataset = HistopathologyDataset(
        data_dir=data_dir,
        magnification=magnification,
        transform=None  # Will set transforms per split
    )
    
    class_info = {
        'class_to_idx': full_dataset.class_to_idx,
        'idx_to_class': full_dataset.idx_to_class,
        'num_classes': len(full_dataset.class_to_idx),
        'num_stages': 5,  # Stages 0-4
        'num_risk_levels': 3  # Low, Medium, High
    }
    
    # Split dataset
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create subset datasets
    from torch.utils.data import Subset
    
    train_dataset = Subset(
        HistopathologyDataset(
            data_dir=data_dir,
            magnification=magnification,
            transform=get_transforms('train', img_size),
            class_to_idx=class_info['class_to_idx']
        ),
        train_indices
    )
    
    val_dataset = Subset(
        HistopathologyDataset(
            data_dir=data_dir,
            magnification=magnification,
            transform=get_transforms('val', img_size),
            class_to_idx=class_info['class_to_idx']
        ),
        val_indices
    )
    
    test_dataset = Subset(
        HistopathologyDataset(
            data_dir=data_dir,
            magnification=magnification,
            transform=get_transforms('test', img_size),
            class_to_idx=class_info['class_to_idx']
        ),
        test_indices
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataset splits:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print(f"  Classes: {class_info['num_classes']}")
    print(f"  Stages: {class_info['num_stages']}")
    
    return train_loader, val_loader, test_loader, class_info
