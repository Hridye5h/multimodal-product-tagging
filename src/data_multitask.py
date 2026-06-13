"""Multi-attribute Fashion data + CLIP-ready dataset.

We keep SEVERAL label columns (articleType,
baseColour, gender, season) and predict them all at once. Images + titles are
fed through a CLIPProcessor.
"""
import os
import glob
import random

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from config import *

def _find(root, name):
    hits = glob.glob(os.path.join(root, "**", name), recursive=True)
    return hits[0] if hits else None

def _find_images_dir(root):
    for h in glob.glob(os.path.join(root, "**", "images"), recursive=True):
        if os.path.isdir(h):
            return h
    return None

def _read_styles(path):
    """Robust read of styles.csv (productDisplayName can contain commas)."""
    with open(path, encoding="utf-8") as f:
        header = [c.strip() for c in f.readline().rstrip("\n").split(",")]
        ncol = len(header)
        recs = [parts for line in f
                if len(parts := line.rstrip("\n").split(",", ncol - 1)) == ncol]
    return pd.DataFrame(recs, columns=header)

def _synthetic_multitag():
    """Fake multi-attribute data so the pipeline runs without the dataset (used when the dataset isn't attached)."""
    rng = random.Random(SEED)
    arts = ["Tshirts", "Shirts", "Watches", "Casual Shoes", "Handbags"]
    cols = ["Black", "Blue", "White", "Red", "Green"]
    gens = ["Men", "Women", "Unisex"]
    seas = ["Summer", "Winter", "Fall"]
    rows = []
    for i in range(SYNTHETIC_N):
        a, c, g, s = rng.choice(arts), rng.choice(cols), rng.choice(gens), rng.choice(seas)
        rows.append({"id": f"syn_{i}", "title": f"{g} {c} {a}",
                     "articleType": a, "baseColour": c, "gender": g, "season": s,
                     "image_path": None})
    df = pd.DataFrame(rows)
    maps = {a: {v: i for i, v in enumerate(sorted(df[a].unique()))} for a in TAG_ATTRIBUTES}
    return df, maps

def build_tagging_df():
    """Load Fashion data keeping all tagged attributes. Returns (df, label_maps, is_synthetic)."""
    styles = images_dir = None
    for root in [DATA_ROOT, "/kaggle/input"]:
        if root and os.path.exists(root):
            styles = styles or _find(root, "styles.csv")
            images_dir = images_dir or _find_images_dir(root)
            if styles and images_dir:
                break

    if not styles or not images_dir:
        print("=" * 72)
        print("  WARNING: Fashion dataset NOT FOUND -> using SYNTHETIC dummy data.")
        print("  Attach 'Fashion Product Images Small' (paramaggarwal) on Kaggle and re-run.")
        print("=" * 72)
        return (*_synthetic_multitag(), True)

    print(f"[data] Loaded real dataset from: {styles}")
    df = _read_styles(styles)
    need = ["id", "productDisplayName"] + TAG_ATTRIBUTES
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"styles.csv missing columns {missing}; have {list(df.columns)}")
    df = df[need].rename(columns={"productDisplayName": "title"})
    for c in ["title"] + TAG_ATTRIBUTES:
        df = df[df[c].str.len() > 0]

    df["image_path"] = df["id"].map(lambda i: os.path.join(images_dir, f"{i}.jpg"))
    df = df[df.image_path.map(os.path.exists)]

    # Keep the top-K most frequent values per attribute (computed on the full data),
    # then drop rows that fall outside any attribute's top-K so every row is fully labelled.
    tops = {a: set(df[a].value_counts().head(TAG_TOPK[a]).index) for a in TAG_ATTRIBUTES}
    for a in TAG_ATTRIBUTES:
        df = df[df[a].isin(tops[a])]
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    maps = {a: {v: i for i, v in enumerate(sorted(df[a].unique()))} for a in TAG_ATTRIBUTES}
    return df, maps, False

def split_df(df):
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n = len(df)
    n_test, n_val = int(n * TEST_FRAC), int(n * VAL_FRAC)
    return df.iloc[n_test + n_val:], df.iloc[n_test:n_test + n_val], df.iloc[:n_test]

class MultiTagDataset(Dataset):
    """Yields {pixel_values, input_ids, attention_mask, labels={attr: idx}} via the CLIP processor."""

    def __init__(self, df, processor, label_maps, train=False, text_dropout=0.0, blank_text=False):
        self.df = df.reset_index(drop=True)
        self.proc = processor
        self.maps = label_maps
        self.attrs = list(label_maps.keys())
        self.text_dropout = text_dropout
        self.blank_text = blank_text

    def __len__(self):
        return len(self.df)

    def _image(self, path):
        try:
            if path and os.path.exists(path):
                return Image.open(path).convert("RGB")
        except Exception:
            pass
        return Image.fromarray((np.random.rand(224, 224, 3) * 255).astype("uint8"))

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = self._image(r.get("image_path", None))
        title = str(r["title"])
        if self.blank_text or (self.text_dropout > 0 and random.random() < self.text_dropout):
            title = ""
        pix = self.proc(images=img, return_tensors="pt")["pixel_values"][0]
        tok = self.proc(text=title, return_tensors="pt", padding="max_length",
                        max_length=TAG_TEXT_MAXLEN, truncation=True)
        labels = {a: torch.tensor(self.maps[a][r[a]], dtype=torch.long) for a in self.attrs}
        return {"pixel_values": pix, "input_ids": tok["input_ids"][0],
                "attention_mask": tok["attention_mask"][0], "labels": labels}
