"""
Utility functions for training, visualization, and model management.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import os
from datetime import datetime


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(
    state: Dict,
    is_best: bool,
    save_dir: str,
    filename: str = 'checkpoint.pth'
):
    """
    Save model checkpoint.
    
    Args:
        state: State dictionary containing model, optimizer, etc.
        is_best: Whether this is the best model so far
        save_dir: Directory to save checkpoint
        filename: Checkpoint filename
    """
    os.makedirs(save_dir, exist_ok=True)
    
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = os.path.join(save_dir, 'best_model.pth')
        torch.save(state, best_filepath)
        print(f"Saved best model to {best_filepath}")


def load_checkpoint(checkpoint_path: str) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
    
    Returns:
        Checkpoint dictionary
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # PyTorch 2.6+ defaults to weights_only=True, which breaks loading
    # checkpoints that contain non-weight objects (like numpy arrays).
    # This project saves full state dicts that we trust, so we explicitly
    # set weights_only=False for compatibility.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return checkpoint


def plot_training_history(
    history: Dict,
    save_path: Optional[str] = None,
    figsize: tuple = (15, 5)
):
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        history: Training history dictionary
        save_path: Path to save the figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot stage accuracy
    axes[1].plot(epochs, history['train_stage_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_stage_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Stage Classification Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot progression accuracy
    axes[2].plot(epochs, history['train_progression_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[2].plot(epochs, history['val_progression_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Accuracy (%)', fontsize=12)
    axes[2].set_title('Progression Risk Accuracy', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.close()


def plot_sample_predictions(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    num_samples: int = 8,
    save_path: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    stage_names: Optional[List[str]] = None
):
    """
    Plot sample predictions with ground truth labels.
    
    Args:
        model: Trained model
        data_loader: Data loader
        device: Device to run inference on
        num_samples: Number of samples to visualize
        save_path: Path to save the figure
        class_names: Names of classes
        stage_names: Names of cancer stages
    """
    if stage_names is None:
        stage_names = ['Stage 0', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']
    
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Get a batch of data
    images, targets = next(iter(data_loader))
    images = images.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        stage_preds = outputs['stage_logits'].argmax(dim=1)
        stage_probs = outputs['stage_probs'].max(dim=1)[0]
    
    # Denormalize images for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    images_denorm = images * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)
    
    # Plot
    num_samples = min(num_samples, len(images))
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_samples):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Display image
        img = images_denorm[idx].cpu().permute(1, 2, 0).numpy()
        ax.imshow(img)
        
        # Get labels
        true_stage = targets['stage'][idx].item()
        pred_stage = stage_preds[idx].item()
        prob = stage_probs[idx].item()
        
        # Title with prediction
        title = f"True: {stage_names[true_stage]}\n"
        title += f"Pred: {stage_names[pred_stage]} ({prob:.2%})"
        
        color = 'green' if true_stage == pred_stage else 'red'
        ax.set_title(title, fontsize=10, color=color, fontweight='bold')
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(num_samples, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample predictions plot saved to {save_path}")
    
    plt.close()


def calculate_class_weights(data_loader: torch.utils.data.DataLoader) -> torch.Tensor:
    """
    Calculate class weights for handling class imbalance.
    
    Args:
        data_loader: Data loader
    
    Returns:
        Class weights tensor
    """
    class_counts = {}
    total_samples = 0
    
    for _, targets in data_loader:
        stages = targets['stage'].numpy()
        for stage in stages:
            class_counts[stage] = class_counts.get(stage, 0) + 1
            total_samples += 1
    
    num_classes = len(class_counts)
    weights = torch.zeros(num_classes)
    
    for class_idx, count in class_counts.items():
        weights[class_idx] = total_samples / (num_classes * count)
    
    return weights


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_results_directory(base_dir: str = 'results') -> str:
    """
    Create a timestamped results directory.
    
    Args:
        base_dir: Base directory for results
    
    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(base_dir, f'run_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir
