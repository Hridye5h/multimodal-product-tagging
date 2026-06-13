"""End-to-end driver for the CLIP multi-task tagging model.

Run from the repo root:  python src/run_multitask.py
On Kaggle, prefer notebooks/multimodal_product_tagging.ipynb.
"""
import os
import json
import random

import numpy as np
import torch
from transformers import CLIPProcessor

from config import SEED, CLIP_MODEL, ARTIFACT_DIR
from data_multitask import build_tagging_df, split_df
from engine_multitask import run_tagging

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    df, label_maps, synth = build_tagging_df()
    print(f"{len(df)} products | "
          + ", ".join(f"{a}={len(m)}" for a, m in label_maps.items())
          + f" | synthetic={synth}")

    train_df, val_df, test_df = split_df(df)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    res, model = run_tagging(train_df, val_df, test_df, processor, label_maps)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, "clip_multitask.pt"))
    with open(os.path.join(ARTIFACT_DIR, "tag_label_maps.json"), "w") as f:
        json.dump({a: {str(k): v for k, v in m.items()} for a, m in label_maps.items()}, f)
    res.to_csv(os.path.join(ARTIFACT_DIR, "tagging_results.csv"), index=False)
    print("\nSaved model + label maps + results to", ARTIFACT_DIR)

if __name__ == "__main__":
    main()
