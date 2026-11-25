"""
Training module for cancer stage prediction model.
Handles training loop, loss computation, and metric tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple, Optional
import os
from datetime import datetime

from src.model import CancerStagePredictor
from src.utils import AverageMeter, save_checkpoint, load_checkpoint


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


class MultiTaskLoss(nn.Module):
    """
    Combined loss for stage classification and progression risk prediction.
    """
    def __init__(
        self,
        stage_weight: float = 1.0,
        progression_weight: float = 0.5,
        use_focal: bool = True,
        alpha: Optional[torch.Tensor] = None
    ):
        super(MultiTaskLoss, self).__init__()
        self.stage_weight = stage_weight
        self.progression_weight = progression_weight
        
        if use_focal:
            self.stage_loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
            self.progression_loss_fn = FocalLoss(alpha=None, gamma=2.0)
        else:
            self.stage_loss_fn = nn.CrossEntropyLoss()
            self.progression_loss_fn = nn.CrossEntropyLoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary with individual loss components
        """
        stage_loss = self.stage_loss_fn(
            outputs['stage_logits'],
            targets['stage']
        )
        
        progression_loss = self.progression_loss_fn(
            outputs['progression_logits'],
            targets['progression_risk']
        )
        
        total_loss = (
            self.stage_weight * stage_loss +
            self.progression_weight * progression_loss
        )
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'stage_loss': stage_loss.item(),
            'progression_loss': progression_loss.item()
        }
        
        return total_loss, loss_dict


def train_epoch(
    model: CancerStagePredictor,
    train_loader: DataLoader,
    criterion: MultiTaskLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Returns:
        Dictionary of training metrics
    """
    model.train()
    
    # Metrics
    losses = AverageMeter()
    stage_losses = AverageMeter()
    progression_losses = AverageMeter()
    
    stage_correct = 0
    progression_correct = 0
    total_samples = 0
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Compute loss
        loss, loss_dict = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        batch_size = images.size(0)
        losses.update(loss_dict['total_loss'], batch_size)
        stage_losses.update(loss_dict['stage_loss'], batch_size)
        progression_losses.update(loss_dict['progression_loss'], batch_size)
        
        # Calculate accuracy
        stage_preds = outputs['stage_logits'].argmax(dim=1)
        progression_preds = outputs['progression_logits'].argmax(dim=1)
        
        stage_correct += (stage_preds == targets['stage']).sum().item()
        progression_correct += (progression_preds == targets['progression_risk']).sum().item()
        total_samples += batch_size
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Stage Acc': f'{100 * stage_correct / total_samples:.2f}%',
            'Prog Acc': f'{100 * progression_correct / total_samples:.2f}%'
        })
    
    metrics = {
        'loss': losses.avg,
        'stage_loss': stage_losses.avg,
        'progression_loss': progression_losses.avg,
        'stage_acc': 100 * stage_correct / total_samples,
        'progression_acc': 100 * progression_correct / total_samples
    }
    
    return metrics


def validate_epoch(
    model: CancerStagePredictor,
    val_loader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """
    Validate for one epoch.
    
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    
    # Metrics
    losses = AverageMeter()
    stage_losses = AverageMeter()
    progression_losses = AverageMeter()
    
    stage_correct = 0
    progression_correct = 0
    total_samples = 0
    
    all_stage_preds = []
    all_stage_targets = []
    all_progression_preds = []
    all_progression_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]')
        
        for images, targets in pbar:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss, loss_dict = criterion(outputs, targets)
            
            # Update metrics
            batch_size = images.size(0)
            losses.update(loss_dict['total_loss'], batch_size)
            stage_losses.update(loss_dict['stage_loss'], batch_size)
            progression_losses.update(loss_dict['progression_loss'], batch_size)
            
            # Calculate accuracy
            stage_preds = outputs['stage_logits'].argmax(dim=1)
            progression_preds = outputs['progression_logits'].argmax(dim=1)
            
            stage_correct += (stage_preds == targets['stage']).sum().item()
            progression_correct += (progression_preds == targets['progression_risk']).sum().item()
            total_samples += batch_size
            
            # Store predictions for detailed metrics
            all_stage_preds.extend(stage_preds.cpu().numpy())
            all_stage_targets.extend(targets['stage'].cpu().numpy())
            all_progression_preds.extend(progression_preds.cpu().numpy())
            all_progression_targets.extend(targets['progression_risk'].cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Stage Acc': f'{100 * stage_correct / total_samples:.2f}%',
                'Prog Acc': f'{100 * progression_correct / total_samples:.2f}%'
            })
    
    metrics = {
        'loss': losses.avg,
        'stage_loss': stage_losses.avg,
        'progression_loss': progression_losses.avg,
        'stage_acc': 100 * stage_correct / total_samples,
        'progression_acc': 100 * progression_correct / total_samples,
        'stage_preds': np.array(all_stage_preds),
        'stage_targets': np.array(all_stage_targets),
        'progression_preds': np.array(all_progression_preds),
        'progression_targets': np.array(all_progression_targets)
    }
    
    return metrics


def train_model(
    model: CancerStagePredictor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    device: str = 'cuda',
    save_dir: str = 'models',
    resume: Optional[str] = None,
    use_focal_loss: bool = True
) -> Tuple[CancerStagePredictor, Dict]:
    """
    Main training function.
    
    Returns:
        Trained model and training history
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss function
    criterion = MultiTaskLoss(
        stage_weight=1.0,
        progression_weight=0.5,
        use_focal=use_focal_loss
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_stage_acc': [],
        'train_progression_acc': [],
        'val_loss': [],
        'val_stage_acc': [],
        'val_progression_acc': []
    }
    
    best_val_acc = 0.0
    start_epoch = 0
    
    # Resume from checkpoint if provided
    if resume and os.path.exists(resume):
        print(f"Resuming from checkpoint: {resume}")
        checkpoint = load_checkpoint(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        history = checkpoint.get('history', history)
    
    print(f"Starting training on device: {device}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(start_epoch, num_epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_stage_acc'].append(train_metrics['stage_acc'])
        history['train_progression_acc'].append(train_metrics['progression_acc'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_stage_acc'].append(val_metrics['stage_acc'])
        history['val_progression_acc'].append(val_metrics['progression_acc'])
        
        # Save checkpoint
        is_best = val_metrics['stage_acc'] > best_val_acc
        if is_best:
            best_val_acc = val_metrics['stage_acc']
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'history': history,
            'val_metrics': val_metrics
        }
        
        save_checkpoint(
            checkpoint,
            is_best,
            save_dir,
            filename=f'checkpoint_epoch_{epoch+1}.pth'
        )
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Stage Acc: {train_metrics['stage_acc']:.2f}%, "
              f"Prog Acc: {train_metrics['progression_acc']:.2f}%")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Stage Acc: {val_metrics['stage_acc']:.2f}%, "
              f"Prog Acc: {val_metrics['progression_acc']:.2f}%")
        print(f"Best Val Acc: {best_val_acc:.2f}%")
        print("-" * 60)
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
    
    return model, history
