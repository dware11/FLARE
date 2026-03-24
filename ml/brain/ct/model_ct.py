"""
CT slice models.

CTSlicedSequenceModel (v2):
- DenseNet121 encodes each axial slice -> fixed-size vector per slice.
- BiLSTM or BiGRU runs over the ordered k vectors (paper-style CNN + sequence).
- Dropout after the RNN reduces overfitting and gradient spikes.
- Per-slice linear classifier produces logits (B, k, num_classes); study-level logits
  combine slices by one of:
  - center: only the middle slice (original behavior, context in RNN state still uses neighbors).
  - max: element-wise max over slices (emphasizes strongest slice evidence; CQ500-style).
  - topk_mean: mean of the top-k slices ranked by abnormal-class logit (focal slices).
- Optional slice thickness (Wang-style): normalized mm concatenated to aggregated logits
  and passed through a small fusion linear layer.

Legacy: DenseNet121 with one classifier head on (B, C, H, W) batches; multi-slice studies
use per-slice logits + mean/max pooling (see patient_logits_legacy).
"""

from typing import Optional, Tuple

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


def _pool_slice_logits(
    slice_logits: torch.Tensor,
    mode: str,
    topk: int,
) -> torch.Tensor:
    """
    slice_logits: (B, k, C) class logits per slice.
    Returns (B, C) study logits.
    """
    b, k, c = slice_logits.shape
    mid = k // 2
    mode = mode.lower()
    if mode == "center":
        return slice_logits[:, mid, :]
    if mode == "max":
        return slice_logits.max(dim=1).values
    if mode == "topk_mean":
        tk = min(max(topk, 1), k)
        # Rank slices by abnormal logit (channel 1)
        scores = slice_logits[:, :, 1]
        _, idx = scores.topk(tk, dim=1)
        expanded = idx.unsqueeze(-1).expand(b, tk, c)
        gathered = torch.gather(slice_logits, 1, expanded)
        return gathered.mean(dim=1)
    raise ValueError(f"Unknown sequence_pool={mode!r}; use center, max, topk_mean")


class CTSlicedSequenceModel(nn.Module):
    """
    Per-slice DenseNet + bidirectional RNN + per-slice logits + study pooling + optional thickness.
    Input: (B, k, C, H, W). Optional thickness: (B, 1) normalized (see dataset).
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        hidden_size: int = 512,
        num_layers: int = 2,
        rnn_type: str = "gru",
        sequence_pool: str = "max",
        topk: int = 3,
        rnn_dropout: float = 0.25,
        use_thickness: bool = False,
    ):
        super().__init__()
        self.backbone, feat_dim = _densenet121_backbone_features(pretrained=pretrained)
        self.feat_dim = feat_dim
        self.sequence_pool = sequence_pool
        self.topk = topk
        self.use_thickness = use_thickness
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
        rnn_out_dim = hidden_size * 2
        self.rnn_dropout = nn.Dropout(rnn_dropout) if rnn_dropout and rnn_dropout > 0 else nn.Identity()
        self.slice_head = nn.Linear(rnn_out_dim, num_classes)
        if use_thickness:
            self.thickness_fusion = nn.Linear(num_classes + 1, num_classes)
        else:
            self.thickness_fusion = None

    def forward(self, x: torch.Tensor, thickness: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected (B, k, C, H, W), got shape {tuple(x.shape)}")
        b, k, c, h, w = x.shape
        flat = x.reshape(b * k, c, h, w)
        feat = self.backbone(flat)
        feat = feat.reshape(b, k, self.feat_dim)
        out, _ = self.rnn(feat)
        out = self.rnn_dropout(out)
        slice_logits = self.slice_head(out)
        logits = _pool_slice_logits(slice_logits, self.sequence_pool, self.topk)
        if self.use_thickness:
            if thickness is None:
                thick = torch.zeros(b, 1, device=x.device, dtype=logits.dtype)
            else:
                thick = thickness.view(b, 1).to(dtype=logits.dtype)
            logits = self.thickness_fusion(torch.cat([logits, thick], dim=-1))
        return logits


def build_ct_sequence_model(
    num_classes: int = 2,
    pretrained: bool = True,
    hidden_size: int = 512,
    num_layers: int = 2,
    rnn_type: str = "gru",
    sequence_pool: str = "max",
    topk: int = 3,
    rnn_dropout: float = 0.25,
    use_thickness: bool = False,
) -> CTSlicedSequenceModel:
    return CTSlicedSequenceModel(
        num_classes=num_classes,
        pretrained=pretrained,
        hidden_size=hidden_size,
        num_layers=num_layers,
        rnn_type=rnn_type,
        sequence_pool=sequence_pool,
        topk=topk,
        rnn_dropout=rnn_dropout,
        use_thickness=use_thickness,
    )


def is_sequence_ct_model(module: nn.Module) -> bool:
    return isinstance(module, CTSlicedSequenceModel)


def load_sequence_weights_compat(model: CTSlicedSequenceModel, state: dict) -> None:
    """
    Load weights from checkpoint. Maps v1 keys (head.*) -> slice_head.* for older checkpoints
    that used center-slice logits only; new checkpoints use slice_head.*.
    """
    if "slice_head.weight" in state:
        model.load_state_dict(state, strict=False)
        return
    remapped = {}
    for k, v in state.items():
        if k.startswith("head."):
            remapped["slice_head." + k[5:]] = v
        else:
            remapped[k] = v
    model.load_state_dict(remapped, strict=False)


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
    thickness: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dispatch: sequence model forward vs legacy per-slice + pool."""
    if is_sequence_ct_model(model):
        return model(x, thickness=thickness)
    return patient_logits_legacy(model, x, agg=agg)
