"""End-to-end driver: data -> ablation -> save artifacts.

Run from the repo root:  python src/run.py
On Kaggle, prefer the notebook (notebooks/multimodal_product_categorization.ipynb).
"""
import os
import json
import random

import numpy as np
import torch
from transformers import AutoTokenizer

from config import SEED, TEXT_MODEL, ARTIFACT_DIR
from data import build_dataframe, make_splits
from engine import run_ablation

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    df, is_synth = build_dataframe()
    print(f"Loaded {len(df)} products / {df.category.nunique()} categories (synthetic={is_synth})")

    train_df, val_df, test_df, label2idx, idx2label = make_splits(df)
    print(f"train/val/test = {len(train_df)}/{len(val_df)}/{len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    results_df, models = run_ablation(train_df, val_df, test_df, tokenizer, label2idx)
    print("\n" + results_df.to_string(index=False))

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    torch.save(models["both"].state_dict(), os.path.join(ARTIFACT_DIR, "fusion_model.pt"))
    with open(os.path.join(ARTIFACT_DIR, "labels.json"), "w") as f:
        json.dump(idx2label, f)
    results_df.to_csv(os.path.join(ARTIFACT_DIR, "ablation_results.csv"), index=False)
    print(f"\nSaved model + labels + results to {ARTIFACT_DIR}/")

if __name__ == "__main__":
    main()
