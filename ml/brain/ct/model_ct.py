import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


def build_ct_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Build DenseNet121 classifier for num_classes (default 2: normal/abnormal).

    DenseNet121 is preferred over ResNet18 for medical CT: dense feature reuse
    improves gradient flow and has strong precedent (CheXNet chest X-ray).
    ImageNet pretrained weights align with the ImageNet normalization applied
    in CTDataset.__getitem__.
    """
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model



