"""CLIP-backed multi-task tagging model.

CLIP gives us *aligned* text and image embeddings out of the box. We concatenate
them, pass through a shared trunk, and branch into one linear head per attribute
(category / colour / gender / season). Frozen CLIP + trained heads is fast and
strong; set FREEZE_CLIP=False to fine-tune.
"""
import torch
import torch.nn as nn
from transformers import CLIPModel

from config import *

class CLIPMultiTask(nn.Module):
    def __init__(self, head_specs, freeze=True, trunk_dim=512, dropout=0.3):
        super().__init__()
        self.frozen = freeze
        self.clip = CLIPModel.from_pretrained(CLIP_MODEL)
        if freeze:
            for p in self.clip.parameters():
                p.requires_grad = False
        d = self.clip.config.projection_dim          # 512 for ViT-B/32
        self.trunk = nn.Sequential(
            nn.LayerNorm(2 * d), nn.Dropout(dropout),
            nn.Linear(2 * d, trunk_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict({a: nn.Linear(trunk_dim, n) for a, n in head_specs.items()})

    def _features(self, batch):
        # Use the full CLIP forward and read the projected embeds. This is stable
        # across transformers versions; get_image_features/get_text_features have
        # drifted (a recent build returns a BaseModelOutputWithPooling, not a tensor).
        out = self.clip(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            return_dict=True,
        )
        return out.image_embeds, out.text_embeds

    def forward(self, batch):
        if self.frozen:
            with torch.no_grad():
                img, txt = self._features(batch)
        else:
            img, txt = self._features(batch)
        h = self.trunk(torch.cat([img, txt], dim=1))
        return {a: head(h) for a, head in self.heads.items()}
