"""Text, image, and fusion encoders.

One `MultiModalModel` class serves all three ablation arms (text-only,
image-only, both) so the comparison is apples-to-apples: same head, same
training, only the inputs differ.
"""
import torch
import torch.nn as nn
import torchvision
from transformers import AutoModel

from config import *

class TextEncoder(nn.Module):
    """DistilBERT -> 768-d [CLS] embedding."""
    out_dim = 768

    def __init__(self, freeze=True):
        super().__init__()
        self.bert = AutoModel.from_pretrained(TEXT_MODEL)
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]  # DistilBERT has no pooler; use [CLS] position

class ImageEncoder(nn.Module):
    """ResNet-50 (ImageNet) -> 2048-d embedding."""
    out_dim = 2048

    def __init__(self, freeze=True):
        super().__init__()
        m = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        )
        m.fc = nn.Identity()
        self.backbone = m
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, image):
        return self.backbone(image)

class MultiModalModel(nn.Module):
    """modality in {'text', 'image', 'both'}. Builds only the encoders it needs."""

    def __init__(self, modality, num_classes, freeze=True, hidden=512, dropout=0.3):
        super().__init__()
        assert modality in {"text", "image", "both"}
        self.modality = modality
        in_dim = 0
        if modality in {"text", "both"}:
            self.text = TextEncoder(freeze)
            in_dim += TextEncoder.out_dim
        if modality in {"image", "both"}:
            self.image = ImageEncoder(freeze)
            in_dim += ImageEncoder.out_dim
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, batch):
        feats = []
        if self.modality in {"text", "both"}:
            feats.append(self.text(batch["input_ids"], batch["attention_mask"]))
        if self.modality in {"image", "both"}:
            feats.append(self.image(batch["image"]))
        x = torch.cat(feats, dim=1) if len(feats) > 1 else feats[0]
        return self.head(x)
