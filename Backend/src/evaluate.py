"""
Evaluation module for model performance assessment.
Computes detailed metrics, confusion matrices, and classification reports.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
import os

from src.model import CancerStagePredictor
from src.utils import load_checkpoint


def evaluate_model(
    model: CancerStagePredictor,
    test_loader: DataLoader,
    device: str = 'cuda',
    class_names: Optional[list] = None,
    stage_names: Optional[list] = None,
    risk_names: Optional[list] = None
) -> Dict:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run evaluation on
        class_names: Names of original classes
        stage_names: Names of cancer stages
        risk_names: Names of progression risk levels
    
    Returns:
        Dictionary containing all evaluation metrics
    """
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    if stage_names is None:
        stage_names = ['Stage 0', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']
    
    if risk_names is None:
        risk_names = ['Low Risk', 'Medium Risk', 'High Risk']
    
    # Storage for predictions and targets
    all_stage_preds = []
    all_stage_targets = []
    all_stage_probs = []
    
    all_progression_preds = []
    all_progression_targets = []
    all_progression_probs = []
    
    all_class_preds = []
    all_class_targets = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            # Forward pass
            outputs = model(images)
            
            # Get predictions
            stage_preds = outputs['stage_logits'].argmax(dim=1)
            progression_preds = outputs['progression_logits'].argmax(dim=1)
            
            # Store results
            all_stage_preds.extend(stage_preds.cpu().numpy())
            all_stage_targets.extend(targets['stage'].cpu().numpy())
            all_stage_probs.extend(outputs['stage_probs'].cpu().numpy())
            
            all_progression_preds.extend(progression_preds.cpu().numpy())
            all_progression_targets.extend(targets['progression_risk'].cpu().numpy())
            all_progression_probs.extend(outputs['progression_probs'].cpu().numpy())
            
            all_class_preds.extend(stage_preds.cpu().numpy())  # Using stage as proxy
            all_class_targets.extend(targets['class'].cpu().numpy())
    
    # Convert to numpy arrays
    all_stage_preds = np.array(all_stage_preds)
    all_stage_targets = np.array(all_stage_targets)
    all_stage_probs = np.array(all_stage_probs)
    
    all_progression_preds = np.array(all_progression_preds)
    all_progression_targets = np.array(all_progression_targets)
    all_progression_probs = np.array(all_progression_probs)
    
    # Calculate metrics for stage classification
    stage_accuracy = accuracy_score(all_stage_targets, all_stage_preds)
    stage_precision = precision_score(
        all_stage_targets, all_stage_preds, average='weighted', zero_division=0
    )
    stage_recall = recall_score(
        all_stage_targets, all_stage_preds, average='weighted', zero_division=0
    )
    stage_f1 = f1_score(
        all_stage_targets, all_stage_preds, average='weighted', zero_division=0
    )
    
    # Calculate metrics for progression risk
    progression_accuracy = accuracy_score(all_progression_targets, all_progression_preds)
    progression_precision = precision_score(
        all_progression_targets, all_progression_preds, average='weighted', zero_division=0
    )
    progression_recall = recall_score(
        all_progression_targets, all_progression_preds, average='weighted', zero_division=0
    )
    progression_f1 = f1_score(
        all_progression_targets, all_progression_preds, average='weighted', zero_division=0
    )
    
    # Confusion matrices
    stage_cm = confusion_matrix(all_stage_targets, all_stage_preds)
    progression_cm = confusion_matrix(all_progression_targets, all_progression_preds)
    
    # Classification reports
    stage_report = classification_report(
        all_stage_targets, all_stage_preds,
        target_names=stage_names,
        output_dict=True,
        zero_division=0
    )
    
    progression_report = classification_report(
        all_progression_targets, all_progression_preds,
        target_names=risk_names,
        output_dict=True,
        zero_division=0
    )
    
    results = {
        'stage': {
            'accuracy': stage_accuracy,
            'precision': stage_precision,
            'recall': stage_recall,
            'f1_score': stage_f1,
            'confusion_matrix': stage_cm,
            'classification_report': stage_report,
            'predictions': all_stage_preds,
            'targets': all_stage_targets,
            'probabilities': all_stage_probs
        },
        'progression': {
            'accuracy': progression_accuracy,
            'precision': progression_precision,
            'recall': progression_recall,
            'f1_score': progression_f1,
            'confusion_matrix': progression_cm,
            'classification_report': progression_report,
            'predictions': all_progression_preds,
            'targets': all_progression_targets,
            'probabilities': all_progression_probs
        }
    }
    
    return results


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    title: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def print_evaluation_summary(results: Dict):
    """
    Print a formatted summary of evaluation results.
    
    Args:
        results: Results dictionary from evaluate_model
    """
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    # Stage classification results
    print("\n📊 CANCER STAGE CLASSIFICATION:")
    print("-" * 70)
    stage_metrics = results['stage']
    print(f"Accuracy:  {stage_metrics['accuracy']:.4f} ({stage_metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {stage_metrics['precision']:.4f}")
    print(f"Recall:    {stage_metrics['recall']:.4f}")
    print(f"F1-Score:  {stage_metrics['f1_score']:.4f}")
    
    # Progression risk results
    print("\n📈 PROGRESSION RISK PREDICTION:")
    print("-" * 70)
    progression_metrics = results['progression']
    print(f"Accuracy:  {progression_metrics['accuracy']:.4f} ({progression_metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {progression_metrics['precision']:.4f}")
    print(f"Recall:    {progression_metrics['recall']:.4f}")
    print(f"F1-Score:  {progression_metrics['f1_score']:.4f}")
    
    print("\n" + "=" * 70)


def save_evaluation_results(
    results: Dict,
    save_dir: str,
    stage_names: Optional[list] = None,
    risk_names: Optional[list] = None
):
    """
    Save evaluation results including confusion matrices and reports.
    
    Args:
        results: Results dictionary from evaluate_model
        save_dir: Directory to save results
        stage_names: Names of cancer stages
        risk_names: Names of progression risk levels
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if stage_names is None:
        stage_names = ['Stage 0', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']
    
    if risk_names is None:
        risk_names = ['Low Risk', 'Medium Risk', 'High Risk']
    
    # Save confusion matrices
    plot_confusion_matrix(
        results['stage']['confusion_matrix'],
        stage_names,
        'Cancer Stage Classification - Confusion Matrix',
        os.path.join(save_dir, 'confusion_matrix_stage.png')
    )
    
    plot_confusion_matrix(
        results['progression']['confusion_matrix'],
        risk_names,
        'Progression Risk Prediction - Confusion Matrix',
        os.path.join(save_dir, 'confusion_matrix_progression.png')
    )
    
    # Save classification reports as text
    with open(os.path.join(save_dir, 'classification_report_stage.txt'), 'w') as f:
        f.write("CANCER STAGE CLASSIFICATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(str(results['stage']['classification_report']))
    
    with open(os.path.join(save_dir, 'classification_report_progression.txt'), 'w') as f:
        f.write("PROGRESSION RISK PREDICTION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(str(results['progression']['classification_report']))
    
    # Save metrics summary
    with open(os.path.join(save_dir, 'metrics_summary.txt'), 'w') as f:
        f.write("EVALUATION METRICS SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write("CANCER STAGE CLASSIFICATION:\n")
        f.write(f"  Accuracy:  {results['stage']['accuracy']:.4f}\n")
        f.write(f"  Precision: {results['stage']['precision']:.4f}\n")
        f.write(f"  Recall:    {results['stage']['recall']:.4f}\n")
        f.write(f"  F1-Score:  {results['stage']['f1_score']:.4f}\n\n")
        f.write("PROGRESSION RISK PREDICTION:\n")
        f.write(f"  Accuracy:  {results['progression']['accuracy']:.4f}\n")
        f.write(f"  Precision: {results['progression']['precision']:.4f}\n")
        f.write(f"  Recall:    {results['progression']['recall']:.4f}\n")
        f.write(f"  F1-Score:  {results['progression']['f1_score']:.4f}\n")
    
    print(f"\nEvaluation results saved to {save_dir}")
