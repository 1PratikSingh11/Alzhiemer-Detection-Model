# model_definition.py

import torch.nn as nn
from torchvision import models

class SafeResNet18(nn.Module):
    """
    ResNet18 with SHAP-safe ReLU (no inplace ops).
    Final FC layer changed to 4 classes for Alzheimer detection.
    """
    def __init__(self, num_classes=4):
        super(SafeResNet18, self).__init__()

        self.resnet18 = models.resnet18(weights=None)

        # Disable inplace to avoid SHAP conflicts
        for m in self.resnet18.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

        # Adjust final layer
        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)
