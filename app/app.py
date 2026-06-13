"""Gradio demo for Hugging Face Spaces: image + title -> top-3 product categories.

Expects artifacts/fusion_model.pt and artifacts/labels.json (produced by the
notebook or `python src/run.py`). For a Space, vendor the src/ files alongside
this app or add the repo root to sys.path.
"""
import os
import sys
import json

import torch
import gradio as gr
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import TEXT_MODEL, MAX_TEXT_LEN, DEVICE
from models import MultiModalModel
from data import make_transforms

ART = os.environ.get("ARTIFACT_DIR", "artifacts")
idx2label = {int(k): v for k, v in json.load(open(os.path.join(ART, "labels.json"))).items()}

tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
model = MultiModalModel("both", num_classes=len(idx2label), freeze=True)
model.load_state_dict(torch.load(os.path.join(ART, "fusion_model.pt"), map_location=DEVICE))
model.to(DEVICE).eval()
_tf = make_transforms(train=False)


@torch.no_grad()
def predict(image, title):
    if image is None:
        return {}
    enc = tokenizer(title or "", truncation=True, max_length=MAX_TEXT_LEN,
                    padding="max_length", return_tensors="pt")
    batch = {
        "input_ids": enc["input_ids"].to(DEVICE),
        "attention_mask": enc["attention_mask"].to(DEVICE),
        "image": _tf(image.convert("RGB")).unsqueeze(0).to(DEVICE),
    }
    probs = model(batch).softmax(1)[0].cpu()
    k = min(3, len(idx2label))
    top = torch.topk(probs, k=k)
    return {idx2label[i.item()]: float(v) for v, i in zip(top.values, top.indices)}


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Product image"),
        gr.Textbox(label="Product title / description"),
    ],
    outputs=gr.Label(num_top_classes=3, label="Predicted category"),
    title="Multimodal Product Categorization",
    description="Text + image fusion (DistilBERT + ResNet-50) trained on the Fashion Product Images dataset.",
)

if __name__ == "__main__":
    demo.launch()
