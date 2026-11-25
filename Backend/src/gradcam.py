"""
Grad-CAM implementation for explainable AI visualization.
Highlights important tissue regions in histopathology images.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import os

from src.model import CancerStagePredictor


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for model interpretability.
    """
    
    def __init__(self, model: CancerStagePredictor, target_layer: Optional[str] = None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Trained model
            target_layer: Name of target layer for Grad-CAM (if None, uses last conv layer)
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        
        # Register hooks
        self.gradients = []
        self.activations = []
        
        # Determine target layer
        if target_layer is None:
            # Find the last convolutional layer in backbone
            if hasattr(self.model.backbone, 'layer4'):
                # ResNet
                self.target_layer = self.model.backbone.layer4[-1].conv3
            else:
                # Try to find last conv layer
                for module in reversed(list(self.model.backbone.modules())):
                    if isinstance(module, torch.nn.Conv2d):
                        self.target_layer = module
                        break
        
        # Register forward and backward hooks
        self.hook_layers()
    
    def hook_layers(self):
        """Register hooks to capture gradients and activations."""
        def backward_hook(module, grad_in, grad_out):
            self.gradients.append(grad_out[0])
        
        def forward_hook(module, input, output):
            self.activations.append(output)
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None,
        task: str = 'stage'
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_image: Input image tensor (1, C, H, W)
            target_class: Target class index (if None, uses predicted class)
            task: 'stage' or 'progression'
        
        Returns:
            Grad-CAM heatmap as numpy array
        """
        self.gradients = []
        self.activations = []
        
        # Forward pass
        output = self.model(input_image)
        
        if task == 'stage':
            logits = output['stage_logits']
            probs = output['stage_probs']
        else:
            logits = output['progression_logits']
            probs = output['progression_probs']
        
        # Get target class
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[-1]
        activations = self.activations[-1]
        
        # Compute weights (global average pooling of gradients)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Generate CAM
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize (detach before converting to numpy to avoid autograd error)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            image: Original image (H, W, 3) in RGB format
            heatmap: Grad-CAM heatmap (H, W)
            alpha: Transparency factor
            colormap: OpenCV colormap
        
        Returns:
            Overlaid image
        """
        # Resize heatmap to match image size
        if heatmap.shape != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Normalize heatmap to 0-255
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlaid = (alpha * heatmap_colored + (1 - alpha) * image).astype(np.uint8)
        
        return overlaid


def visualize_gradcam(
    model: CancerStagePredictor,
    image_path: str,
    device: str = 'cuda',
    save_path: Optional[str] = None,
    task: str = 'stage',
    target_class: Optional[int] = None,
    img_size: int = 224
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate and visualize Grad-CAM for a single image.
    
    Args:
        model: Trained model
        image_path: Path to input image
        device: Device to run inference on
        save_path: Path to save visualization
        task: 'stage' or 'progression'
        target_class: Target class for Grad-CAM
        img_size: Image size for preprocessing
    
    Returns:
        Tuple of (original image, heatmap, overlaid image)
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Load and preprocess image
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(img_tensor)
        if task == 'stage':
            pred_class = output['stage_logits'].argmax(dim=1).item()
            pred_prob = output['stage_probs'][0, pred_class].item()
        else:
            pred_class = output['progression_logits'].argmax(dim=1).item()
            pred_prob = output['progression_probs'][0, pred_class].item()
    
    # Generate Grad-CAM
    gradcam = GradCAM(model, target_layer=None)
    heatmap = gradcam.generate_cam(img_tensor, target_class=target_class, task=task)
    
    # Convert image to numpy for overlay
    img_np = np.array(img.resize((img_size, img_size)))
    
    # Overlay heatmap
    overlaid = gradcam.overlay_heatmap(img_np, heatmap, alpha=0.4)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(overlaid)
    pred_label = f'Predicted: Class {pred_class} ({pred_prob:.2%})'
    axes[2].set_title(f'Overlaid\n{pred_label}', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grad-CAM visualization saved to {save_path}")
    
    plt.close()
    
    return img_np, heatmap, overlaid


def batch_gradcam_visualization(
    model: CancerStagePredictor,
    image_paths: List[str],
    device: str = 'cuda',
    save_dir: str = 'results/gradcam',
    task: str = 'stage',
    img_size: int = 224
):
    """
    Generate Grad-CAM visualizations for multiple images.
    
    Args:
        model: Trained model
        image_paths: List of image paths
        device: Device to run inference on
        save_dir: Directory to save visualizations
        task: 'stage' or 'progression'
        img_size: Image size for preprocessing
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for i, img_path in enumerate(image_paths):
        save_path = os.path.join(save_dir, f'gradcam_{task}_{i+1}.png')
        try:
            visualize_gradcam(
                model, img_path, device, save_path, task, target_class=None, img_size=img_size
            )
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
