from typing import Tuple

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


def _densenet121_backbone_features(pretrained: bool = True) -> Tuple[nn.Module, int]:
    """DenseNet121 with classifier replaced by Identity; returns (module, feature_dim)."""
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = densenet121(weights=weights)
    feat_dim = backbone.classifier.in_features
    backbone.classifier = nn.Identity()
    return backbone, feat_dim


class CTSlicedSequenceModel(nn.Module):
    """
    Per-slice DenseNet encoder + bidirectional LSTM over k slices, then classify from the
    center time step (paper-style 2.5D / CNN–LSTM). Expects input (B, k, C, H, W).
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        hidden_size: int = 512,
        num_layers: int = 2,
        rnn_type: str = "lstm",
    ):
        super().__init__()
        self.backbone, feat_dim = _densenet121_backbone_features(pretrained=pretrained)
        self.feat_dim = feat_dim
        rnn_type = rnn_type.lower()
        if rnn_type == "gru":
            self.rnn = nn.GRU(
                feat_dim,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
            )
        else:
            self.rnn = nn.LSTM(
                feat_dim,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
            )
        self.head = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected (B, k, C, H, W), got shape {tuple(x.shape)}")
        b, k, c, h, w = x.shape
        flat = x.reshape(b * k, c, h, w)
        feat = self.backbone(flat)
        feat = feat.reshape(b, k, self.feat_dim)
        out, _ = self.rnn(feat)
        center = out[:, k // 2, :]
        return self.head(center)


def build_ct_sequence_model(
    num_classes: int = 2,
    pretrained: bool = True,
    hidden_size: int = 512,
    num_layers: int = 2,
    rnn_type: str = "lstm",
) -> CTSlicedSequenceModel:
    return CTSlicedSequenceModel(
        num_classes=num_classes,
        pretrained=pretrained,
        hidden_size=hidden_size,
        num_layers=num_layers,
        rnn_type=rnn_type,
    )


def is_sequence_ct_model(module: nn.Module) -> bool:
    return isinstance(module, CTSlicedSequenceModel)


def patient_logits_legacy(
    model: nn.Module,
    x: torch.Tensor,
    agg: str = "max",
) -> torch.Tensor:
    """
    x: (B, k, C, H, W). Runs 2D classifier per slice and aggregates to (B, num_classes).
    agg: 'mean' or 'max' over slices (max matches common CQ500 / slice-pooling practice).
    """
    b, k, c, h, w = x.shape
    flat = x.reshape(b * k, c, h, w)
    logits_flat = model(flat)
    logits_per = logits_flat.view(b, k, -1)
    if agg == "mean":
        return logits_per.mean(dim=1)
    return logits_per.max(dim=1).values


def patient_logits_from_model(
    model: nn.Module,
    x: torch.Tensor,
    agg: str = "max",
) -> torch.Tensor:
    """Dispatch: sequence model forward vs legacy per-slice + pool."""
    if is_sequence_ct_model(model):
        return model(x)
    return patient_logits_legacy(model, x, agg=agg)
