# Multimodal Product Tagging

Predicting e-commerce product attributes from a product's **image and title**, on
the [Fashion Product Images](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
dataset (~44k products). Two notebooks:

- **`multimodal_product_tagging.ipynb`** — a single **CLIP** model that tags four
  attributes at once (category, colour, gender, season).
- **`multimodal_product_categorization.ipynb`** — a DistilBERT + ResNet-50 fusion
  model for category classification, with a text-only / image-only / fusion ablation.

Both share a theme: product titles often name the attribute outright, so text alone
scores high. The more interesting question is **robustness** — when the title is
missing (as it often is in real catalogs), does the model fall back on the image?

📊 **[Results & write-up →](https://hridye5h.github.io/multimodal-product-tagging/)**

## Results

**CLIP multi-task tagging** — one model, four attributes, 30k products:

| Attribute | Classes | With title | Title removed |
|-----------|--------:|-----------:|--------------:|
| articleType (category) | 20 | 0.978 | 0.927 |
| baseColour | 15 | 0.935 | 0.744 |
| gender | 5 | 0.991 | 0.893 |
| season | 4 | 0.785 | 0.710 |
| **Mean** | — | **0.922** | **0.819** |

The model only loses ~10 points of mean accuracy when titles are removed — it leans
on CLIP's image features instead of collapsing.

**Single-modality ablation** (category, 20 classes) makes the point sharply:

| Model | With title | Title removed |
|-------|-----------:|--------------:|
| Text-only (DistilBERT) | 0.975 | 0.028 |
| Image-only (ResNet-50) | 0.905 | 0.905 |
| **Fusion** | **0.980** | **0.886** |

When titles vanish, text-only collapses to ~3% while fusion holds ~89% — a +86-point
robustness gain from using the image. The fusion model is trained with random
"missing title" augmentation so it learns to use the image as a fallback.

## Repo layout

```
├── notebooks/
│   ├── multimodal_product_tagging.ipynb          # CLIP, multi-task
│   └── multimodal_product_categorization.ipynb   # fusion + ablation
├── src/
│   ├── config.py             # hyperparameters & paths
│   ├── data.py               # Fashion parsing + dataset
│   ├── models.py             # text / image / fusion encoders
│   ├── engine.py             # training, evaluation, ablation
│   ├── data_multitask.py     # multi-attribute dataset (CLIP)
│   ├── clip_model.py         # CLIP + one head per attribute
│   ├── engine_multitask.py   # multi-task training & evaluation
│   ├── run.py · run_multitask.py
├── app/app.py                # Gradio demo
├── docs/index.html           # results page
└── requirements.txt
```

## Running it

On **Kaggle** (recommended): open a notebook, enable GPU + Internet, add the
*Fashion Product Images (Small)* dataset, and Run All. The loaders find `styles.csv`
and the `images/` folder automatically.

Locally:

```bash
pip install -r requirements.txt
python src/run_multitask.py     # CLIP multi-task tagging
python src/run.py               # fusion + ablation
```

If the dataset isn't present, a small synthetic set lets the pipeline run end to end.

## Method notes

- **CLIP** (`openai/clip-vit-base-patch32`) supplies aligned image/text embeddings;
  a shared trunk feeds one linear head per attribute. Frozen CLIP + trained heads is
  fast and strong; `FREEZE_CLIP=False` fine-tunes.
- **Fusion** concatenates a DistilBERT `[CLS]` embedding with ResNet-50 features,
  trained with discriminative learning rates (fast head, gentle backbone).
- **Robustness** is measured by blanking the title at evaluation; the fusion/tagging
  models train with title dropout so they don't depend on text being present.
