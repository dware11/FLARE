import torch 
import torch.nn as nn 
from torchvision.models import densenet121, DenseNet121_Weights 

def build_ct_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Build DenseNet121 with final classifier for num_classes (default 2: normal/abnormal)."""
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = densenet121(weights=weights)
    in_features = model.classifier.in_features 
    model.classifier = nn.Linear(in_features, num_classes)
    return model 



