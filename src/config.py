"""Central configuration for the multimodal product categorization project.

Every tunable lives here so the notebook, the CLI runner, and the Gradio app
all read the same knobs. The defaults are tuned for a *fast first run* on a
Kaggle GPU (frozen backbones, subsampled data); flip the marked flags for the
full training run that produces the full results.
"""
import os
import torch

# ---- Reproducibility ----
SEED = 42

# ---- Data: Fashion Product Images (Kaggle) ----
# One click on Kaggle: Add Data -> "Fashion Product Images (Small)"
# (paramaggarwal/fashion-product-images-small). The loader recursively finds
# styles.csv + the images/ folder, so the exact mount nesting doesn't matter.
DATA_ROOT = os.environ.get("DATA_ROOT", "/kaggle/input/fashion-product-images-small")
LABEL_COL = "articleType"    # what to predict. "articleType" (many, harder, best
                             # fusion story), "subCategory" (~45), "masterCategory" (~7, easy).
TOP_K_CATEGORIES = 20        # keep the K most frequent classes
MAX_PER_CLASS = 1500         # subsample each class -> fast first run
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# If the dataset isn't found, generate a tiny synthetic set so the whole pipeline
# still runs end-to-end (the numbers are meaningless). Handy for checking the code
# path on a laptop before attaching the real dataset.
USE_SYNTHETIC_FALLBACK = True
SYNTHETIC_N = 1200

# ---- Text encoder ----
TEXT_MODEL = "distilbert-base-uncased"
MAX_TEXT_LEN = 48

# ---- Image encoder ----
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---- Training ----
# Defaults do a real FINE-TUNING run (unfrozen encoders) for strong results.
# For a quick check instead, set FREEZE_BACKBONE = True (trains only the head; weak numbers).
BATCH_SIZE = 32          # fits two fine-tuned backbones on a 16 GB T4; drop to 16 if you hit OOM
EPOCHS = 5
FREEZE_BACKBONE = False
HEAD_LR = 1e-3           # classifier head learns fast
BACKBONE_LR = 2e-5       # encoders fine-tune gently (unused when FREEZE_BACKBONE=True)
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2

# Robustness experiment: product titles often spell out the category, so text alone
# nearly maxes out. To show the IMAGE earns its keep, we train the fusion model with
# random "missing title" augmentation, then test every arm with titles present vs.
# removed. Fusion should stay strong without titles (it falls back on the image),
# while text-only collapses.
TEXT_DROPOUT = 0.5       # fraction of titles blanked during fusion training

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "artifacts")

# ---- CLIP multi-task tagging ----
# One model reads image + title and predicts several catalog attributes
# at once (category, colour, gender, season) using CLIP's aligned embeddings.
CLIP_MODEL = "openai/clip-vit-base-patch32"
FREEZE_CLIP = True            # frozen CLIP features are strong AND fast; unfreeze for a final push
TAG_ATTRIBUTES = ["articleType", "baseColour", "gender", "season"]
TAG_TOPK = {"articleType": 20, "baseColour": 15, "gender": 5, "season": 5}
TAG_TEXT_MAXLEN = 32
TAG_HEAD_LR = 1e-3            # trunk + per-attribute heads learn fast
TAG_CLIP_LR = 1e-5           # CLIP fine-tunes gently (unused when FREEZE_CLIP=True)
TAG_EPOCHS = 6
