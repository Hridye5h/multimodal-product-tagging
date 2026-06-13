"""Fashion Product Images (Kaggle) parsing + the PyTorch dataset.

The dataset ships a `styles.csv` (product metadata incl. productDisplayName and
category columns) plus an `images/<id>.jpg` folder. We join them into clean
(title, category, image_path) rows, keep the top-K categories, and subsample.

If the dataset isn't present we fall back to a small synthetic set so the rest
of the pipeline still runs (see config.USE_SYNTHETIC_FALLBACK).
"""
import os
import glob
import random

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from config import *

def _find_file(root, name):
    hits = glob.glob(os.path.join(root, "**", name), recursive=True)
    return hits[0] if hits else None

def _find_images_dir(root):
    for h in glob.glob(os.path.join(root, "**", "images"), recursive=True):
        if os.path.isdir(h):
            return h
    return None

def load_fashion(data_root):
    """Return DataFrame[title, category, image_path] from the Fashion Product Images dataset."""
    styles = _find_file(data_root, "styles.csv")
    images_dir = _find_images_dir(data_root)
    if not styles or not images_dir:
        return pd.DataFrame(columns=["title", "category", "image_path"])
    # productDisplayName (the last column) can contain unescaped commas, so a naive
    # CSV read drops those rows. Split with a capped maxsplit so the final field
    # absorbs any extra commas -> we keep every row instead of skipping them.
    with open(styles, encoding="utf-8") as f:
        header = [c.strip() for c in f.readline().rstrip("\n").split(",")]
        ncol = len(header)
        recs = [parts for line in f
                if len(parts := line.rstrip("\n").split(",", ncol - 1)) == ncol]
    df = pd.DataFrame(recs, columns=header)
    if LABEL_COL not in df.columns or "productDisplayName" not in df.columns:
        return pd.DataFrame(columns=["title", "category", "image_path"])
    df = df[["id", "productDisplayName", LABEL_COL]]
    df = df[(df["productDisplayName"].str.len() > 0) & (df[LABEL_COL].str.len() > 0)]
    df["image_path"] = df["id"].map(lambda i: os.path.join(images_dir, f"{i}.jpg"))
    df = df[df.image_path.map(os.path.exists)]
    df = df.rename(columns={"productDisplayName": "title", LABEL_COL: "category"})
    return df[["title", "category", "image_path"]]

def _synthetic_dataframe():
    """Tiny fake dataset: titles literally contain the category, images are noise.
    Text-only will score well, image-only ~chance, fusion ~text. quick check only."""
    cats = [f"CATEGORY_{i}" for i in range(min(TOP_K_CATEGORIES, 8))]
    words = ["cotton", "steel", "wooden", "wireless", "leather", "ceramic", "plastic",
             "cordless", "vintage", "compact", "premium", "portable", "stainless", "ergonomic"]
    rng = random.Random(SEED)
    rows = []
    for i in range(SYNTHETIC_N):
        c = rng.choice(cats)
        title = " ".join(rng.sample(words, 4)) + f" {c.lower()}"
        rows.append((f"syn_{i}", title, c, None))
    return pd.DataFrame(rows, columns=["image_id", "title", "category", "image_path"])

def build_dataframe():
    """Load + clean the dataset, keep top-K categories, subsample. Returns (df, is_synthetic)."""
    df = pd.DataFrame()
    for root in [DATA_ROOT, "/kaggle/input"]:   # try the configured path, then any attached dataset
        if root and os.path.exists(root):
            df = load_fashion(root)
            if len(df):
                print(f"[data] Loaded real dataset from: {root}")
                break

    if len(df) == 0:
        if not USE_SYNTHETIC_FALLBACK:
            raise RuntimeError(
                f"No dataset found under {DATA_ROOT}. Attach 'Fashion Product Images (Small)' "
                f"or set USE_SYNTHETIC_FALLBACK=True."
            )
        print("=" * 72)
        print("  WARNING: Fashion dataset NOT FOUND  ->  using SYNTHETIC dummy data.")
        print("  The numbers below are MEANINGLESS (text=100% by construction, image=chance).")
        print("  Fix on Kaggle:  Add Data (right panel) -> search")
        print("  'Fashion Product Images Small' (by paramaggarwal) -> Add -> Run All again.")
        print("=" * 72)
        return _synthetic_dataframe(), True

    top = df.category.value_counts().head(TOP_K_CATEGORIES).index
    df = df[df.category.isin(top)]
    # Subsample each class via explicit concat (robust across pandas versions;
    # groupby.apply can fold the grouping column into the index).
    parts = [g.sample(min(len(g), MAX_PER_CLASS), random_state=SEED)
             for _, g in df.groupby("category")]
    df = pd.concat(parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return df, False

def make_splits(df):
    """Shuffle and carve into train/val/test. Returns (train, val, test, label2idx, idx2label)."""
    labels = sorted(df.category.unique())
    label2idx = {c: i for i, c in enumerate(labels)}
    idx2label = {i: c for c, i in label2idx.items()}
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n = len(df)
    n_test = int(n * TEST_FRAC)
    n_val = int(n * VAL_FRAC)
    test = df.iloc[:n_test]
    val = df.iloc[n_test:n_test + n_val]
    train = df.iloc[n_test + n_val:]
    return train, val, test, label2idx, idx2label

def make_transforms(train):
    if train:
        return T.Compose([
            T.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize(int(IMAGE_SIZE * 1.14)),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

class ProductDataset(Dataset):
    """Yields {input_ids, attention_mask, image, label} for one product."""

    def __init__(self, df, tokenizer, label2idx, train=False, text_dropout=0.0, blank_text=False):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.label2idx = label2idx
        self.tf = make_transforms(train)
        self.text_dropout = text_dropout   # train-time: randomly blank this fraction of titles
        self.blank_text = blank_text        # eval-time: blank every title (simulate missing metadata)

    def __len__(self):
        return len(self.df)

    def _load_image(self, path):
        try:
            if path and os.path.exists(path):
                return Image.open(path).convert("RGB")
        except Exception:
            pass
        # Synthetic / missing image -> random noise (keeps the transform path uniform).
        arr = (np.random.rand(IMAGE_SIZE, IMAGE_SIZE, 3) * 255).astype("uint8")
        return Image.fromarray(arr)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        image = self.tf(self._load_image(r.get("image_path", None)))
        title = str(r["title"])
        # Blank the title when simulating missing metadata (eval) or for training-time
        # dropout. An empty string still tokenizes to [CLS][SEP] -> a valid "no text" input.
        if self.blank_text or (self.text_dropout > 0 and random.random() < self.text_dropout):
            title = ""
        enc = self.tok(
            title, truncation=True, max_length=MAX_TEXT_LEN,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "image": image,
            "label": torch.tensor(self.label2idx[r["category"]], dtype=torch.long),
        }
