"""Lightweight explainability for the deepfake face classifiers.

Grad-CAM: EfficientNet-B0's last convolutional feature block.
Occlusion sensitivity: architecture-agnostic batched patch masking.
Heatmaps are normalized per image; they indicate sensitivity, not causal proof.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_heatmap(values: torch.Tensor) -> np.ndarray:
    values = values.detach().float().cpu().numpy()
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - low) / (high - low)).astype(np.float32)


def efficientnet_target_layer(model):
    """timm EfficientNet: conv_head is the final convolution before BN/pooling."""
    if not hasattr(model, 'conv_head'):
        raise ValueError('Grad-CAM target is only configured for EfficientNet-B0.')
    return model.conv_head


def gradcam_efficientnet(model, input_tensor: torch.Tensor, class_index: int) -> np.ndarray:
    """Compute Grad-CAM for the selected output class without retaining gradients afterward."""
    activations = {}
    gradients = {}

    def forward_hook(_module, _inputs, output):
        activations['value'] = output

    def backward_hook(_module, _grad_input, grad_output):
        gradients['value'] = grad_output[0]

    target = efficientnet_target_layer(model)
    forward_handle = target.register_forward_hook(forward_hook)
    backward_handle = target.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        output = model(input_tensor)
        score = output[0, class_index]
        score.backward()
        features = activations['value']
        grads = gradients['value']
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * features).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False)
        return normalize_heatmap(cam[0, 0])
    finally:
        forward_handle.remove()
        backward_handle.remove()
        model.zero_grad(set_to_none=True)


def occlusion_sensitivity(
    model,
    input_tensor: torch.Tensor,
    class_index: int,
    grid_size: int = 32,
) -> np.ndarray:
    """Measure probability drop when masking each cell of an N x N grid.

    Each batch contains the original followed by all masked variants, so the
    baseline uses exactly the same inference path. Mask value is ImageNet mean,
    which becomes zero after normalization.
    """
    if grid_size not in (16, 32, 64):
        raise ValueError('grid_size must be 16, 32, or 64')
    if input_tensor.shape[-2:] != (224, 224):
        raise ValueError('Expected 224x224 model input')

    cell = 224 // grid_size
    variants = [input_tensor]
    for row in range(grid_size):
        for col in range(grid_size):
            masked = input_tensor.clone()
            y1, y2 = row * cell, (row + 1) * cell
            x1, x2 = col * cell, (col + 1) * cell
            masked[..., y1:y2, x1:x2] = -IMAGENET_MEAN / IMAGENET_STD
            variants.append(masked)
    batch = torch.cat(variants, dim=0)
    with torch.inference_mode():
        logits = model(batch)
        probabilities = torch.softmax(logits.float(), dim=1)[:, class_index]
    baseline = probabilities[0]
    importance = baseline - probabilities[1:]
    # Positive values mean masking lowers the selected class score.
    importance = importance.reshape(grid_size, grid_size).unsqueeze(0).unsqueeze(0)
    importance = F.interpolate(importance, size=(224, 224), mode='bilinear', align_corners=False)
    return normalize_heatmap(importance[0, 0])


def heatmap_overlay(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.42) -> Image.Image:
    """Render an RGB JET-like heatmap overlay using Pillow/Matplotlib only."""
    from matplotlib import colormaps

    image = image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
    rgba = colormaps['jet'](np.clip(heatmap, 0, 1))[..., :3]
    rgba = (rgba * 255).astype(np.uint8)
    base = np.asarray(image).astype(np.float32)
    mixed = (1 - alpha) * base + alpha * rgba.astype(np.float32)
    return Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8))


def heatmap_image(heatmap: np.ndarray) -> Image.Image:
    from matplotlib import colormaps

    rgba = colormaps['jet'](np.clip(heatmap, 0, 1))
    return Image.fromarray((rgba * 255).astype(np.uint8), mode='RGBA').convert('RGB')
