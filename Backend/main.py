"""
Main entry point for the AI Tool for Automatic Cancer Stage and Progression Prediction.

This script orchestrates the complete pipeline:
1. Data loading and preprocessing
2. Model training
3. Model evaluation
4. Grad-CAM visualization
5. Results generation

Usage:
    python main.py --mode train --data_dir <path_to_dataset>
    python main.py --mode evaluate --model_path <path_to_model>
    python main.py --mode gradcam --model_path <path_to_model> --image_path <path_to_image>
"""

import argparse
import os
import torch
import numpy as np
from pathlib import Path

from src.dataset import create_data_loaders
from src.model import create_model
from src.train import train_model
from src.evaluate import evaluate_model, print_evaluation_summary, save_evaluation_results
from src.gradcam import visualize_gradcam, batch_gradcam_visualization
from src.utils import (
    set_seed, plot_training_history, plot_sample_predictions,
    create_results_directory, load_checkpoint
)


def train_pipeline(args):
    """Execute training pipeline."""
    print("=" * 70)
    print("TRAINING PIPELINE")
    print("=" * 70)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create results directory
    results_dir = create_results_directory(args.results_dir)
    print(f"Results will be saved to: {results_dir}")
    
    # Create data loaders
    print("\n📂 Loading dataset...")
    train_loader, val_loader, test_loader, class_info = create_data_loaders(
        data_dir=args.data_dir,
        magnification=args.magnification,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_split=args.train_split,
        val_split=args.val_split,
        img_size=args.img_size,
        seed=args.seed
    )
    
    # Create model
    print("\n🏗️  Creating model...")
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    model = create_model(
        num_stages=class_info['num_stages'],
        num_risk_levels=class_info['num_risk_levels'],
        backbone=args.backbone,
        pretrained=args.pretrained,
        device=device
    )
    
    # Train model
    print("\n🚀 Starting training...")
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        save_dir=os.path.join(results_dir, 'checkpoints'),
        resume=args.resume,
        use_focal_loss=args.use_focal_loss
    )
    
    # Plot training history
    print("\n📊 Generating training plots...")
    plot_training_history(
        history,
        save_path=os.path.join(results_dir, 'training_history.png')
    )
    
    # Evaluate on test set
    print("\n📈 Evaluating on test set...")
    test_results = evaluate_model(
        model=trained_model,
        test_loader=test_loader,
        device=device,
        stage_names=['Stage 0', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4'],
        risk_names=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    print_evaluation_summary(test_results)
    save_evaluation_results(
        test_results,
        save_dir=os.path.join(results_dir, 'evaluation'),
        stage_names=['Stage 0', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4'],
        risk_names=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    # Generate sample predictions visualization
    print("\n🖼️  Generating sample predictions...")
    plot_sample_predictions(
        model=trained_model,
        data_loader=test_loader,
        device=device,
        num_samples=8,
        save_path=os.path.join(results_dir, 'sample_predictions.png'),
        stage_names=['Stage 0', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']
    )
    
    print(f"\n✅ Training pipeline completed! Results saved to: {results_dir}")


def evaluate_pipeline(args):
    """Execute evaluation pipeline."""
    print("=" * 70)
    print("EVALUATION PIPELINE")
    print("=" * 70)
    
    # Set seed
    set_seed(args.seed)
    
    # Create results directory
    results_dir = create_results_directory(args.results_dir)
    
    # Load checkpoint
    print(f"\n📂 Loading model from: {args.model_path}")
    checkpoint = load_checkpoint(args.model_path)
    
    # Create data loaders
    print("\n📂 Loading dataset...")
    train_loader, val_loader, test_loader, class_info = create_data_loaders(
        data_dir=args.data_dir,
        magnification=args.magnification,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_split=args.train_split,
        val_split=args.val_split,
        img_size=args.img_size,
        seed=args.seed
    )
    
    # Create and load model
    print("\n🏗️  Creating model...")
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    model = create_model(
        num_stages=class_info['num_stages'],
        num_risk_levels=class_info['num_risk_levels'],
        backbone=args.backbone,
        pretrained=False,
        device=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    print("\n📈 Evaluating model...")
    test_results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        stage_names=['Stage 0', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4'],
        risk_names=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    print_evaluation_summary(test_results)
    save_evaluation_results(
        test_results,
        save_dir=os.path.join(results_dir, 'evaluation'),
        stage_names=['Stage 0', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4'],
        risk_names=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    # Generate sample predictions
    plot_sample_predictions(
        model=model,
        data_loader=test_loader,
        device=device,
        num_samples=8,
        save_path=os.path.join(results_dir, 'sample_predictions.png'),
        stage_names=['Stage 0', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']
    )
    
    print(f"\n✅ Evaluation completed! Results saved to: {results_dir}")


def gradcam_pipeline(args):
    """Execute Grad-CAM visualization pipeline."""
    print("=" * 70)
    print("GRAD-CAM VISUALIZATION PIPELINE")
    print("=" * 70)
    
    # Create results directory
    results_dir = create_results_directory(args.results_dir)
    gradcam_dir = os.path.join(results_dir, 'gradcam')
    os.makedirs(gradcam_dir, exist_ok=True)
    
    # Load checkpoint
    print(f"\n📂 Loading model from: {args.model_path}")
    checkpoint = load_checkpoint(args.model_path)
    
    # Create data loaders to get class info
    _, _, _, class_info = create_data_loaders(
        data_dir=args.data_dir,
        magnification=args.magnification,
        batch_size=1,
        num_workers=0,
        train_split=0.7,
        val_split=0.15,
        img_size=args.img_size,
        seed=args.seed
    )
    
    # Create and load model
    print("\n🏗️  Creating model...")
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    model = create_model(
        num_stages=class_info['num_stages'],
        num_risk_levels=class_info['num_risk_levels'],
        backbone=args.backbone,
        pretrained=False,
        device=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Generate Grad-CAM for single image or batch
    if args.image_path:
        if os.path.isfile(args.image_path):
            # Single image
            print(f"\n🖼️  Generating Grad-CAM for: {args.image_path}")
            save_path = os.path.join(gradcam_dir, 'gradcam_stage.png')
            visualize_gradcam(
                model=model,
                image_path=args.image_path,
                device=device,
                save_path=save_path,
                task='stage',
                img_size=args.img_size
            )
            
            # Also generate for progression risk
            save_path = os.path.join(gradcam_dir, 'gradcam_progression.png')
            visualize_gradcam(
                model=model,
                image_path=args.image_path,
                device=device,
                save_path=save_path,
                task='progression',
                img_size=args.img_size
            )
        elif os.path.isdir(args.image_path):
            # Directory of images
            print(f"\n🖼️  Generating Grad-CAM for images in: {args.image_path}")
            image_files = [
                os.path.join(args.image_path, f)
                for f in os.listdir(args.image_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
            ][:args.num_samples]
            
            batch_gradcam_visualization(
                model=model,
                image_paths=image_files,
                device=device,
                save_dir=gradcam_dir,
                task='stage',
                img_size=args.img_size
            )
    
    print(f"\n✅ Grad-CAM visualization completed! Results saved to: {gradcam_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='AI Tool for Automatic Cancer Stage and Progression Prediction'
    )
    
    # General arguments
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'evaluate', 'gradcam'],
                       help='Pipeline mode: train, evaluate, or gradcam')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage (even if CUDA available)')
    
    # Data arguments
    parser.add_argument('--magnification', type=str, default='400X',
                       choices=['40X', '100X', '200X', '400X'],
                       help='Microscopy magnification level')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training/evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--train_split', type=float, default=0.7,
                       help='Fraction of data for training')
    parser.add_argument('--val_split', type=float, default=0.15,
                       help='Fraction of data for validation')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='resnet50',
                       choices=['resnet50', 'efficientnet'],
                       help='Backbone architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for optimizer')
    parser.add_argument('--use_focal_loss', action='store_true', default=True,
                       help='Use focal loss for class imbalance')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Evaluation arguments
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model checkpoint for evaluation/gradcam')
    
    # Grad-CAM arguments
    parser.add_argument('--image_path', type=str, default=None,
                       help='Path to image or directory for Grad-CAM')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples for batch Grad-CAM')
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode in ['evaluate', 'gradcam']:
        if args.model_path is None:
            parser.error(f"--model_path is required for mode '{args.mode}'")
    
    if args.mode == 'gradcam':
        if args.image_path is None:
            parser.error("--image_path is required for mode 'gradcam'")
    
    # Execute appropriate pipeline
    if args.mode == 'train':
        train_pipeline(args)
    elif args.mode == 'evaluate':
        evaluate_pipeline(args)
    elif args.mode == 'gradcam':
        gradcam_pipeline(args)


if __name__ == '__main__':
    main()
