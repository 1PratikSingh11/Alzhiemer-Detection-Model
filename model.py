"""
SafeResNet-18 Architecture for Alzheimer's Disease MRI Classification.

This module implements SafeResNet18, matching the exact trained architecture
from the OASIS final report, with all ReLU layers recursively set to inplace=False
for full compatibility with SHAP GradientExplainer and autograd backward hooks.
"""

from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import torchvision.models as models

# 4 Clinical Severity Classes from OASIS Dataset (matching model output order)
CLASS_NAMES: List[str] = [
    "Non Demented",
    "Very mild Dementia",
    "Mild Dementia",
    "Moderate Dementia"
]


class SafeResNet18(nn.Module):
    """
    SafeResNet-18 Model for Alzheimer's Disease MRI Classification.
    
    Features:
      - Backbone: ResNet-18 with 4 residual stages and skip connections.
      - Safe Activations: All ReLUs recursively converted to inplace=False.
      - Classification Head: 512-dimensional embedding mapped to 4 disease classes (512 -> 4).
      - Hooks for feature map extraction (for Grad-CAM comparison module).
    """

    def __init__(self, num_classes: int = 4, pretrained: bool = False):
        super(SafeResNet18, self).__init__()
        self.num_classes = num_classes
        self.class_names = CLASS_NAMES

        # Initialize ResNet-18 backbone
        self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        # Disable in-place ReLU operations to avoid SHAP GradientExplainer autograd errors
        for m in self.resnet18.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

        # Adjust final classification layer: 512 -> 4 classes
        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized logits."""
        return self.resnet18(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts the 512-dimensional feature embedding before the linear layer."""
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)

        x = self.resnet18.avgpool(x)
        return torch.flatten(x, 1)

    def get_last_conv_layer(self) -> nn.Module:
        """Returns the final convolutional layer of layer4 (for Grad-CAM)."""
        return self.resnet18.layer4[-1].conv2

    def load_weights(self, checkpoint_path: str, device: torch.device) -> None:
        """Loads state_dict handling both 'resnet18.' prefixed and unprefixed keys."""
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        # Check if keys have 'resnet18.' prefix
        sample_key = next(iter(state_dict.keys()))
        if sample_key.startswith("resnet18."):
            self.load_state_dict(state_dict)
        elif sample_key.startswith("resnet."):
            # Map resnet. -> resnet18.
            remapped = {k.replace("resnet.", "resnet18.", 1): v for k, v in state_dict.items()}
            self.load_state_dict(remapped)
        else:
            # Wrap in resnet18.
            remapped = {f"resnet18.{k}": v for k, v in state_dict.items()}
            self.load_state_dict(remapped)


def build_model(
    num_classes: int = 4,
    pretrained: bool = False,
    dropout_rate: float = 0.0
) -> SafeResNet18:
    """Factory helper to construct SafeResNet18."""
    return SafeResNet18(num_classes=num_classes, pretrained=pretrained)
