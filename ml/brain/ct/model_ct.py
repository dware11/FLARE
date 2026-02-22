import torch 
import torch.nn as nn 
from torchvision.models import resnet18, ResNet18_Weights 

def build_ct_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Build ResNet18 with final FC for num_classes (default 2: normal/abnormal)."""
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights) 
    model.fc = nn.Linear(model.fc.in_features, num_classes) 
    return model 



