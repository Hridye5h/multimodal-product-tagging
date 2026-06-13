"""Multi-task training / evaluation + the robustness test.

Loss is the sum of per-attribute cross-entropies. We report each attribute's
accuracy with the title present vs. removed — the multimodal model should hold up
on visual attributes (colour) even when the title is gone.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import *
from data_multitask import MultiTagDataset
from clip_model import CLIPMultiTask

def _to_device(batch):
    out = {}
    for k, v in batch.items():
        out[k] = {a: t.to(DEVICE) for a, t in v.items()} if k == "labels" else v.to(DEVICE)
    return out

def _loader(ds, shuffle=False):
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS)

@torch.no_grad()
def evaluate_multitask(model, loader, attrs):
    model.eval()
    correct = {a: 0 for a in attrs}
    total = 0
    for batch in loader:
        batch = _to_device(batch)
        out = model(batch)
        total += next(iter(out.values())).size(0)
        for a in attrs:
            correct[a] += (out[a].argmax(1) == batch["labels"][a]).sum().item()
    return {a: correct[a] / max(total, 1) for a in attrs}

def train_multitask(model, train_ds, val_ds, attrs):
    train_loader, val_loader = _loader(train_ds, shuffle=True), _loader(val_ds)
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("clip")]
    clip_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("clip")]
    groups = [{"params": head_params, "lr": TAG_HEAD_LR}]
    if clip_params:
        groups.append({"params": clip_params, "lr": TAG_CLIP_LR})
    opt = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()

    best_mean, best_state = -1.0, None
    for epoch in range(1, TAG_EPOCHS + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = _to_device(batch)
            opt.zero_grad()
            out = model(batch)
            loss = sum(crit(out[a], batch["labels"][a]) for a in attrs)
            loss.backward()
            opt.step()
            running += loss.item()
        accs = evaluate_multitask(model, val_loader, attrs)
        mean_acc = float(np.mean(list(accs.values())))
        print(f"epoch {epoch}/{TAG_EPOCHS}  loss {running / max(len(train_loader), 1):.3f}  "
              f"val_mean_acc {mean_acc:.3f}  " + "  ".join(f"{a[:4]}={accs[a]:.2f}" for a in attrs))
        if mean_acc > best_mean:
            best_mean = mean_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model

def run_tagging(train_df, val_df, test_df, processor, label_maps):
    attrs = list(label_maps.keys())
    head_specs = {a: len(label_maps[a]) for a in attrs}
    train_ds = MultiTagDataset(train_df, processor, label_maps, train=True, text_dropout=TEXT_DROPOUT)
    val_ds = MultiTagDataset(val_df, processor, label_maps)
    test_full = MultiTagDataset(test_df, processor, label_maps)
    test_blank = MultiTagDataset(test_df, processor, label_maps, blank_text=True)

    print(f"Training CLIP multi-task model (frozen={FREEZE_CLIP}) on {attrs} ...")
    model = CLIPMultiTask(head_specs, freeze=FREEZE_CLIP).to(DEVICE)
    model = train_multitask(model, train_ds, val_ds, attrs)

    acc_full = evaluate_multitask(model, _loader(test_full), attrs)
    acc_blank = evaluate_multitask(model, _loader(test_blank), attrs)
    res = pd.DataFrame([{
        "attribute": a, "n_classes": head_specs[a],
        "acc_with_title": round(acc_full[a], 4),
        "acc_no_title": round(acc_blank[a], 4),
    } for a in attrs])

    print("\nPer-attribute accuracy (one CLIP model, four heads):")
    print(res.to_string(index=False))
    print(f"\nMean accuracy with title:    {np.mean(list(acc_full.values())):.3f}")
    print(f"Mean accuracy, title removed: {np.mean(list(acc_blank.values())):.3f}  "
          "(visual attributes like colour survive on the image alone)")
    return res, model
