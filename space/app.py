"""Gradio demo: tag a product (category, colour, gender, season) from image + title.

Loads CLIP fresh from Hugging Face and the trained heads (downloaded from the repo),
so the Space stays small. Works even with no title — the image carries the prediction.
"""
import os
import json
import urllib.request

import torch
import torch.nn as nn
import gradio as gr
from transformers import CLIPModel, CLIPProcessor

CLIP_MODEL = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RAW = "https://raw.githubusercontent.com/Hridye5h/multimodal-product-tagging/main/space"

# Pull the trained weights, label maps, and example images on first start.
EXAMPLES = ["tshirt.jpg", "watch.jpg", "shoes.jpg", "handbag.jpg"]
os.makedirs("examples", exist_ok=True)
for fname in ["heads.pt", "tag_label_maps.json"] + [f"examples/{e}" for e in EXAMPLES]:
    if not os.path.exists(fname):
        urllib.request.urlretrieve(f"{RAW}/{fname}", fname)

maps = json.load(open("tag_label_maps.json"))
attrs = list(maps.keys())                                   # articleType, baseColour, gender, season
idx2label = {a: {int(i): lbl for lbl, i in m.items()} for a, m in maps.items()}
head_specs = {a: len(maps[a]) for a in attrs}


class CLIPMultiTask(nn.Module):
    def __init__(self, head_specs, trunk_dim=512, dropout=0.3):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(CLIP_MODEL)
        for p in self.clip.parameters():
            p.requires_grad = False
        d = self.clip.config.projection_dim
        self.trunk = nn.Sequential(
            nn.LayerNorm(2 * d), nn.Dropout(dropout),
            nn.Linear(2 * d, trunk_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict({a: nn.Linear(trunk_dim, n) for a, n in head_specs.items()})

    def forward(self, batch):
        out = self.clip(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"], return_dict=True)
        h = self.trunk(torch.cat([out.image_embeds, out.text_embeds], dim=1))
        return {a: head(h) for a, head in self.heads.items()}


model = CLIPMultiTask(head_specs).to(DEVICE).eval()
model.load_state_dict(torch.load("heads.pt", map_location=DEVICE), strict=False)  # trunk+heads; CLIP from pretrained
proc = CLIPProcessor.from_pretrained(CLIP_MODEL)


@torch.no_grad()
def predict(image, title):
    if image is None:
        return {}, {}, {}, {}
    pix = proc(images=image.convert("RGB"), return_tensors="pt")["pixel_values"].to(DEVICE)
    tok = proc(text=title or "", return_tensors="pt", padding="max_length", max_length=32, truncation=True)
    batch = {"pixel_values": pix,
             "input_ids": tok["input_ids"].to(DEVICE),
             "attention_mask": tok["attention_mask"].to(DEVICE)}
    out = model(batch)
    res = []
    for a in attrs:
        probs = out[a].softmax(1)[0].cpu()
        k = min(5, len(idx2label[a]))
        top = torch.topk(probs, k)
        res.append({idx2label[a][i.item()]: float(v) for v, i in zip(top.values, top.indices)})
    return res[0], res[1], res[2], res[3]


with gr.Blocks(title="Multimodal Product Tagger") as demo:
    gr.Markdown(
        """# 🏷️ Multimodal Product Tagger
Upload a product image and (optionally) its title. One CLIP-based model predicts
**four catalog attributes at once** — category, colour, gender, season.

ℹ️ Works best on **clean catalog-style product photos** (single item, plain background) —
that's the data it was trained on. **Click an example below** to see it work. Real-world
selfies are out-of-distribution and will be off.

Try it **with the title blank** too: the model falls back on the image, so visual
attributes like colour and category still come through."""
    )
    with gr.Row():
        with gr.Column():
            img = gr.Image(type="pil", label="Product image")
            txt = gr.Textbox(label="Product title (optional)",
                             placeholder="e.g. Nike Men Blue Running Shoes")
            btn = gr.Button("Tag it", variant="primary")
        with gr.Column():
            o1 = gr.Label(label="Category", num_top_classes=5)
            o2 = gr.Label(label="Colour", num_top_classes=5)
            o3 = gr.Label(label="Gender", num_top_classes=5)
            o4 = gr.Label(label="Season", num_top_classes=4)
    btn.click(predict, [img, txt], [o1, o2, o3, o4])
    gr.Examples(
        examples=[[f"examples/{e}", ""] for e in EXAMPLES],
        inputs=[img, txt],
        label="Catalog examples — click one, then press “Tag it”",
    )

demo.launch()
