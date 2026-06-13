"""Train / evaluate loops and the ablation runner.

`run_ablation` trains the three arms in turn and returns a tidy results table —
that table (fusion beating both single-modality baselines) is the headline.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from config import *
from data import ProductDataset
from models import MultiModalModel

def _to_device(batch):
    return {k: v.to(DEVICE) for k, v in batch.items()}

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = _to_device(batch)
        logits = model(batch)
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(batch["label"].cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return accuracy_score(y, p), f1_score(y, p, average="macro")

def train_model(modality, train_ds, val_ds, num_classes):
    model = MultiModalModel(modality, num_classes, freeze=FREEZE_BACKBONE).to(DEVICE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    # Discriminative learning rates: the fresh classifier head learns fast, while the
    # pretrained encoders fine-tune gently so we don't wash out their ImageNet/BERT priors.
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if name.startswith("head") else backbone_params).append(p)
    groups = [{"params": head_params, "lr": HEAD_LR}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": BACKBONE_LR})
    opt = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()

    best_f1, best_state = -1.0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = _to_device(batch)
            opt.zero_grad()
            loss = crit(model(batch), batch["label"])
            loss.backward()
            opt.step()
            running += loss.item()
        acc, f1 = evaluate(model, val_loader)
        print(f"[{modality:5}] epoch {epoch}/{EPOCHS}  "
              f"loss {running / max(len(train_loader), 1):.3f}  "
              f"val_acc {acc:.3f}  val_f1 {f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model

def run_ablation(train_df, val_df, test_df, tokenizer, label2idx):
    num_classes = len(label2idx)
    val_ds = ProductDataset(val_df, tokenizer, label2idx, train=False)
    test_full = ProductDataset(test_df, tokenizer, label2idx, train=False)
    test_blank = ProductDataset(test_df, tokenizer, label2idx, train=False, blank_text=True)

    def _acc(model, ds):
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        return evaluate(model, loader)[0]

    results, models = [], {}
    for modality in ["text", "image", "both"]:
        print(f"\n=== Training arm: {modality} ===")
        # Fusion trains with text dropout so it learns to fall back on the image when the
        # title is missing; the single-modality baselines train on clean data.
        td = TEXT_DROPOUT if modality == "both" else 0.0
        train_ds = ProductDataset(train_df, tokenizer, label2idx, train=True, text_dropout=td)
        model = train_model(modality, train_ds, val_ds, num_classes)
        acc_full = _acc(model, test_full)
        # Blanking the title is meaningless for the image-only model -> reuse its score.
        acc_blank = acc_full if modality == "image" else _acc(model, test_blank)
        results.append({"modality": modality,
                        "acc_full_title": round(acc_full, 4),
                        "acc_no_title": round(acc_blank, 4)})
        models[modality] = model
        print(f"--> {modality}: full-title {acc_full:.3f}  |  no-title {acc_blank:.3f}")

    res = pd.DataFrame(results)
    # Headline: when titles vanish, text collapses but fusion holds (it uses the image).
    t = res.set_index("modality")
    gain = (t.loc["both", "acc_no_title"] - t.loc["text", "acc_no_title"]) * 100
    print(f"\nWith titles:    text {t.loc['text','acc_full_title']:.3f}   fusion {t.loc['both','acc_full_title']:.3f}")
    print(f"Titles removed: text {t.loc['text','acc_no_title']:.3f}   fusion {t.loc['both','acc_no_title']:.3f}"
          f"   -> fusion is +{gain:.0f} pts more robust")
    return res, models
