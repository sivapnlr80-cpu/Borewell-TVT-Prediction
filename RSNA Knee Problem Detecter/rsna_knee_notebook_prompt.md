# RSNA Knee Abnormality Detection — Complete Kaggle Notebook Prompt

## Competition Context (for the AI generating the notebook)

You are building a **complete, error-free Kaggle notebook** for the
**RSNA Knee Abnormality Detection 2026** competition hosted on Kaggle by the
Radiological Society of North America (RSNA).

**Competition URL**: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection

---

## Competition Summary

| Field | Details |
|---|---|
| **Task** | Detect knee abnormalities from multimodal data (MRI images + radiology report text) |
| **Data** | 5,000+ knee MRI exams from 16 global institutions; reports in 9+ languages |
| **Imaging format** | DICOM (MRI series — likely axial/sagittal PD, PD-FS, T1, T2 sequences) |
| **Text** | Radiology reports (multilingual: English + 8 other languages) |
| **Label** | Binary or multi-label abnormality flags per study |
| **Evaluation metric** | **ROC AUC Score** (area under the receiver-operating-characteristic curve) |
| **Submission format** | CSV: one row per study with predicted probability per abnormality class |
| **Competition type** | Research Code Competition (notebook must run inside Kaggle environment) |
| **Prize** | $77,000 |
| **Timeline** | ~3 months remaining (ends ~November 2026) |

---

## Notebook Requirements

### Hard Rules (do not violate)
1. The notebook must run **completely from top to bottom without any errors** in the Kaggle environment (Python 3.10+, GPU T4/P100).
2. All imports must be from libraries **available in the Kaggle Docker image** or installable via `!pip install --quiet` at the top.
3. The final cell must write a valid **`submission.csv`** to `/kaggle/working/submission.csv`.
4. Submission CSV must have exactly the columns the competition requires — check `/kaggle/input/rsna-knee-abnormality-detection/sample_submission.csv` and match its exact column names and dtypes.
5. All predictions must be **probabilities in [0, 1]**, not hard labels.
6. No data leakage: train only on `/kaggle/input/rsna-knee-abnormality-detection/train/` data and labels.
7. Internet access is OFF in submission mode — all model weights must be loaded from `/kaggle/input/` (pre-downloaded model datasets) or from local paths.

### Recommended Library Stack (from awesome-python + Kaggle defaults)

**Core scientific**
```python
import numpy as np           # numpy — fundamental array computing
import pandas as pd          # pandas — DataFrames for metadata/labels
from scipy import stats      # scipy — statistical utilities
```

**Medical imaging (DICOM)**
```python
import pydicom               # pydicom — read DICOM files (pre-installed on Kaggle)
import cv2                   # opencv — image resizing, normalization (pre-installed)
from PIL import Image        # pillow — image I/O
```

**Deep learning**
```python
import torch                 # pytorch — primary DL framework
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm                  # pytorch-image-models — pretrained CNNs/ViTs (pip install timm)
```

**NLP / multilingual text (for radiology reports)**
```python
from transformers import (   # huggingface transformers — multilingual BERT/XLM-R
    AutoTokenizer,
    AutoModel,
    XLMRobertaTokenizer,
    XLMRobertaModel,
)
```

**Training utilities**
```python
from sklearn.metrics import roc_auc_score   # scikit-learn — AUC evaluation
from sklearn.model_selection import StratifiedKFold
import albumentations as A                  # albumentations — image augmentation (pip install albumentations)
from albumentations.pytorch import ToTensorV2
```

**Visualization**
```python
import matplotlib.pyplot as plt  # matplotlib
import seaborn as sns            # seaborn
```

---

## Notebook Structure to Generate

Generate a single Jupyter notebook (`rsna_knee_solution.ipynb`) with the following
cells in exact order. Each cell must be complete and runnable.

---

### Cell 1 — Install dependencies

```python
# Install non-default packages quietly
!pip install -q timm albumentations pydicom transformers accelerate
```

---

### Cell 2 — Imports & Config

```python
import os, gc, glob, random, warnings
import numpy as np
import pandas as pd
import cv2
import pydicom
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
from transformers import AutoTokenizer, AutoModel

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore")

# ── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = "/kaggle/input/rsna-knee-abnormality-detection"
TRAIN_DIR  = f"{BASE_DIR}/train"
TEST_DIR   = f"{BASE_DIR}/test"
OUTPUT_DIR = "/kaggle/working"

# ── Hyper-parameters ────────────────────────────────────────────────────────
CFG = dict(
    img_size      = 224,
    batch_size    = 16,
    n_epochs      = 5,
    lr            = 2e-4,
    weight_decay  = 1e-2,
    n_folds       = 5,
    device        = "cuda" if torch.cuda.is_available() else "cpu",
    model_name    = "efficientnet_b4",        # timm backbone for images
    text_model    = "xlm-roberta-base",       # HuggingFace model for reports
    max_text_len  = 256,
    mixed_prec    = True,                     # fp16 AMP
    num_workers   = 2,
)
print("Device:", CFG["device"])
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

---

### Cell 3 — Explore data files

```python
# Print all available files
for root, dirs, files in os.walk(BASE_DIR):
    level = root.replace(BASE_DIR, "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    if level < 2:
        subindent = " " * 2 * (level + 1)
        for f in files[:10]:
            print(f"{subindent}{f}")

# Load metadata / labels
train_df = pd.read_csv(f"{BASE_DIR}/train.csv")
test_df  = pd.read_csv(f"{BASE_DIR}/test.csv")
sample_sub = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")

print("\ntrain_df shape:", train_df.shape)
print(train_df.head())
print("\ntest_df shape:", test_df.shape)
print("\nsample_submission columns:", sample_sub.columns.tolist())
print(sample_sub.head())

# Identify target columns (all columns that are prediction targets)
LABEL_COLS = [c for c in sample_sub.columns if c not in ["study_id", "row_id", "id"]]
print("\nTarget label columns:", LABEL_COLS)
```

---

### Cell 4 — DICOM utilities

```python
def load_dicom_as_array(path: str, img_size: int = CFG["img_size"]) -> np.ndarray:
    """Read a DICOM file, apply VOI LUT, rescale to uint8, resize."""
    dcm = pydicom.dcmread(path)
    img = dcm.pixel_array.astype(np.float32)

    # Apply RescaleSlope / RescaleIntercept if present
    if hasattr(dcm, "RescaleSlope"):
        img = img * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)

    # Normalize to 0-255
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min) * 255.0
    img = img.astype(np.uint8)

    # Convert to 3-channel (some DICOM are grayscale)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[-1] != 3:
        img = img[..., :3]

    img = cv2.resize(img, (img_size, img_size))
    return img


def get_study_dicom_paths(study_id: str, split: str = "train") -> list:
    """Return all DICOM file paths for a given study."""
    folder = TRAIN_DIR if split == "train" else TEST_DIR
    pattern = os.path.join(folder, str(study_id), "**", "*.dcm")
    paths = glob.glob(pattern, recursive=True)
    return sorted(paths)


# Quick sanity check
if len(train_df) > 0:
    sample_study = train_df["study_id"].iloc[0]
    dcm_paths = get_study_dicom_paths(sample_study, "train")
    print(f"Study {sample_study}: {len(dcm_paths)} DICOM files")
    if dcm_paths:
        img = load_dicom_as_array(dcm_paths[0])
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.title(f"Sample slice — study {sample_study}")
        plt.axis("off")
        plt.show()
```

---

### Cell 5 — Dataset class (multimodal: image + text)

```python
# ── Augmentations ────────────────────────────────────────────────────────────
def get_transforms(mode: str = "train"):
    if mode == "train":
        return A.Compose([
            A.RandomResizedCrop(CFG["img_size"], CFG["img_size"], scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.GaussianBlur(blur_limit=(3, 5)),
                A.MotionBlur(blur_limit=3),
            ], p=0.3),
            A.RandomBrightnessContrast(p=0.4),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(CFG["img_size"], CFG["img_size"]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


# ── Tokenizer (lazy-init inside dataset) ─────────────────────────────────────
_TOKENIZER = None
def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(CFG["text_model"])
    return _TOKENIZER


class KneeDataset(Dataset):
    """
    Returns one item per study.
    Image tensor: mean of middle N slices → (3, H, W).
    Text tensor: tokenized radiology report.
    """
    def __init__(self, df: pd.DataFrame, split: str = "train", n_slices: int = 5):
        self.df       = df.reset_index(drop=True)
        self.split    = split
        self.n_slices = n_slices
        self.tfms     = get_transforms(split)
        self.tok      = get_tokenizer()

        # Identify text column (report field)
        text_cols = [c for c in df.columns if "report" in c.lower() or "text" in c.lower()]
        self.text_col = text_cols[0] if text_cols else None

        # Label columns (only for train/val)
        self.label_cols = LABEL_COLS if split != "test" else []

    def __len__(self):
        return len(self.df)

    def _load_image(self, study_id):
        paths = get_study_dicom_paths(str(study_id), self.split)
        if not paths:
            # fallback: blank image
            return torch.zeros(3, CFG["img_size"], CFG["img_size"])

        # Select middle slices
        mid = len(paths) // 2
        half = self.n_slices // 2
        selected = paths[max(0, mid - half): mid + half + 1]
        selected = selected[:self.n_slices]  # cap

        arrays = []
        for p in selected:
            try:
                arr = load_dicom_as_array(p)
                aug = self.tfms(image=arr)["image"]  # (3, H, W) tensor
                arrays.append(aug)
            except Exception:
                continue

        if not arrays:
            return torch.zeros(3, CFG["img_size"], CFG["img_size"])
        return torch.stack(arrays).mean(0)  # average over slices

    def _encode_text(self, text: str):
        if not isinstance(text, str) or text.strip() == "":
            text = "no report available"
        enc = self.tok(
            text,
            max_length=CFG["max_text_len"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_id = row["study_id"]

        img_tensor = self._load_image(study_id)

        text = row[self.text_col] if self.text_col and self.text_col in row else ""
        input_ids, attn_mask = self._encode_text(str(text))

        item = {
            "image":       img_tensor,
            "input_ids":   input_ids,
            "attn_mask":   attn_mask,
            "study_id":    study_id,
        }

        if self.label_cols:
            labels = torch.tensor(
                row[self.label_cols].values.astype(np.float32),
                dtype=torch.float32,
            )
            item["labels"] = labels

        return item
```

---

### Cell 6 — Multimodal model

```python
class ImageEncoder(nn.Module):
    def __init__(self, model_name: str = CFG["model_name"], pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.out_dim  = self.backbone.num_features

    def forward(self, x):
        return self.backbone(x)   # (B, out_dim)


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = CFG["text_model"], pretrained: bool = True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name) if pretrained else \
                       AutoModel.from_pretrained(model_name)
        self.out_dim = self.encoder.config.hidden_size

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token representation
        return out.last_hidden_state[:, 0, :]   # (B, hidden_size)


class MultimodalKneeModel(nn.Module):
    """
    Fusion of image CNN + multilingual text transformer.
    Outputs per-label logits for binary classification.
    """
    def __init__(self, n_classes: int):
        super().__init__()
        self.image_enc = ImageEncoder()
        self.text_enc  = TextEncoder()

        fusion_dim = self.image_enc.out_dim + self.text_enc.out_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, n_classes),
        )

    def forward(self, image, input_ids, attn_mask):
        img_feat  = self.image_enc(image)                        # (B, d_img)
        text_feat = self.text_enc(input_ids, attn_mask)          # (B, d_txt)
        fused     = torch.cat([img_feat, text_feat], dim=-1)     # (B, d_img+d_txt)
        logits    = self.fusion(fused)                           # (B, n_classes)
        return logits


# Quick model test (no forward pass, just init check)
n_classes = len(LABEL_COLS) if LABEL_COLS else 1
print(f"Number of output classes: {n_classes}")
model_test = MultimodalKneeModel(n_classes=n_classes)
total_params = sum(p.numel() for p in model_test.parameters()) / 1e6
print(f"Total parameters: {total_params:.1f}M")
del model_test; gc.collect()
```

---

### Cell 7 — Training loop

```python
from torch.cuda.amp import GradScaler, autocast

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        img   = batch["image"].to(device)
        ids   = batch["input_ids"].to(device)
        mask  = batch["attn_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        with autocast(enabled=CFG["mixed_prec"]):
            logits = model(img, ids, mask)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        img    = batch["image"].to(device)
        ids    = batch["input_ids"].to(device)
        mask   = batch["attn_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast(enabled=CFG["mixed_prec"]):
            logits = model(img, ids, mask)
            loss   = criterion(logits, labels)

        total_loss += loss.item()
        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Per-class AUC, then macro average
    aucs = []
    for i in range(all_labels.shape[1]):
        if len(np.unique(all_labels[:, i])) > 1:
            aucs.append(roc_auc_score(all_labels[:, i], all_preds[:, i]))
    macro_auc = np.mean(aucs) if aucs else 0.0

    return total_loss / len(loader), macro_auc, all_preds


def run_fold(fold: int, train_df: pd.DataFrame, val_df: pd.DataFrame):
    print(f"\n{'='*50}")
    print(f"FOLD {fold}")
    print(f"{'='*50}")

    train_ds = KneeDataset(train_df, split="train")
    val_ds   = KneeDataset(val_df,   split="val")

    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"], shuffle=True,
        num_workers=CFG["num_workers"], pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG["batch_size"] * 2, shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=True
    )

    model = MultimodalKneeModel(n_classes=len(LABEL_COLS)).to(CFG["device"])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["n_epochs"], eta_min=1e-6
    )
    scaler = GradScaler(enabled=CFG["mixed_prec"])

    best_auc   = 0.0
    best_preds = None
    best_path  = os.path.join(OUTPUT_DIR, f"best_model_fold{fold}.pth")

    for epoch in range(1, CFG["n_epochs"] + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, CFG["device"])
        vl_loss, vl_auc, val_preds = validate(model, val_loader, criterion, CFG["device"])
        scheduler.step()

        print(f"Epoch {epoch}/{CFG['n_epochs']} | "
              f"Train Loss: {tr_loss:.4f} | Val Loss: {vl_loss:.4f} | Val AUC: {vl_auc:.4f}")

        if vl_auc > best_auc:
            best_auc   = vl_auc
            best_preds = val_preds.copy()
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ New best AUC: {best_auc:.4f} → saved to {best_path}")

    del model; gc.collect(); torch.cuda.empty_cache()
    return best_preds, val_df["study_id"].values, best_auc
```

---

### Cell 8 — Cross-validation training

```python
# ── Stratified K-Fold on primary label ────────────────────────────────────
skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=SEED)

# Use first label column for stratification; if multi-label, use max
primary_label = train_df[LABEL_COLS[0]] if LABEL_COLS else pd.Series(np.zeros(len(train_df)))

oof_preds  = np.zeros((len(train_df), len(LABEL_COLS)))
fold_aucs  = []

for fold, (tr_idx, vl_idx) in enumerate(skf.split(train_df, primary_label)):
    tr_fold = train_df.iloc[tr_idx]
    vl_fold = train_df.iloc[vl_idx]
    preds, study_ids, auc = run_fold(fold, tr_fold, vl_fold)
    oof_preds[vl_idx] = preds
    fold_aucs.append(auc)
    print(f"Fold {fold} best AUC: {auc:.4f}")

print(f"\n{'='*50}")
print(f"OOF Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
print(f"{'='*50}")

# OOF AUC (overall)
oof_aucs = []
for i, col in enumerate(LABEL_COLS):
    if len(np.unique(train_df[col].values)) > 1:
        oof_aucs.append(roc_auc_score(train_df[col].values, oof_preds[:, i]))
print(f"Overall OOF AUC (macro): {np.mean(oof_aucs):.4f}")
```

---

### Cell 9 — Inference on test set

```python
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_preds = []
    for batch in loader:
        img  = batch["image"].to(device)
        ids  = batch["input_ids"].to(device)
        mask = batch["attn_mask"].to(device)
        with autocast(enabled=CFG["mixed_prec"]):
            logits = model(img, ids, mask)
        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(preds)
    return np.vstack(all_preds)


test_ds = KneeDataset(test_df, split="test")
test_loader = DataLoader(
    test_ds, batch_size=CFG["batch_size"] * 2, shuffle=False,
    num_workers=CFG["num_workers"], pin_memory=True
)

test_preds_all = np.zeros((len(test_df), len(LABEL_COLS)))

for fold in range(CFG["n_folds"]):
    model = MultimodalKneeModel(n_classes=len(LABEL_COLS)).to(CFG["device"])
    ckpt_path = os.path.join(OUTPUT_DIR, f"best_model_fold{fold}.pth")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=CFG["device"]))
        fold_preds = predict(model, test_loader, CFG["device"])
        test_preds_all += fold_preds / CFG["n_folds"]
        print(f"Fold {fold} inference done")
    else:
        print(f"Fold {fold} checkpoint not found, skipping")
    del model; gc.collect(); torch.cuda.empty_cache()

# Clip to [0, 1] just in case
test_preds_all = np.clip(test_preds_all, 0.0, 1.0)
print("Test predictions shape:", test_preds_all.shape)
```

---

### Cell 10 — Generate submission.csv

```python
# ── Build submission matching sample_submission.csv exactly ───────────────
sample_sub = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")
print("Sample submission format:")
print(sample_sub.head())
print("Columns:", sample_sub.columns.tolist())

submission = pd.DataFrame()

# Map study_id column
id_col = "study_id"  # adjust if competition uses a different key
submission[id_col] = test_df[id_col].values

# Fill prediction columns
for i, col in enumerate(LABEL_COLS):
    submission[col] = test_preds_all[:, i]

# Reindex to match sample_submission column order
submission = submission[sample_sub.columns]

# Verify shape and dtypes
assert len(submission) == len(sample_sub), \
    f"Row count mismatch: {len(submission)} vs {len(sample_sub)}"
assert list(submission.columns) == list(sample_sub.columns), \
    f"Column mismatch: {list(submission.columns)} vs {list(sample_sub.columns)}"

# Check no NaN / out-of-range values
for col in LABEL_COLS:
    assert submission[col].notna().all(), f"NaN found in {col}"
    assert (submission[col] >= 0).all() and (submission[col] <= 1).all(), \
        f"Out-of-range predictions in {col}"

# Save
sub_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(sub_path, index=False)
print(f"\nSubmission saved → {sub_path}")
print(f"Shape: {submission.shape}")
print(submission.head())
print(f"\nPrediction stats:")
print(submission[LABEL_COLS].describe())
```

---

## Additional Instructions for the AI Generating This Notebook

1. **Read `sample_submission.csv` first** — do not assume column names. Dynamically detect `LABEL_COLS` from it, not from assumptions.

2. **Handle missing text** — radiology reports may be null/empty; always supply a fallback string (`"no report"`) to avoid tokenizer errors.

3. **Handle missing DICOMs** — some studies may have no DICOM files; return a zero tensor instead of crashing.

4. **Avoid hardcoded paths** — always construct paths from `BASE_DIR` constants.

5. **Memory management** — call `gc.collect()` and `torch.cuda.empty_cache()` after each fold to avoid OOM.

6. **`autocast` context** — always wrap forward passes in `torch.cuda.amp.autocast` to benefit from fp16 and avoid dtype mismatches.

7. **Pretrained model source** — in Kaggle's no-internet submission mode, load pretrained weights from a pre-attached Kaggle dataset (e.g., `timm` weights from `/kaggle/input/timm-efficientnet-weights/`). Add a helper that falls back to random initialization if weights are unavailable.

8. **Multilingual text** — use `xlm-roberta-base` (XLM-R) as it handles all 9+ report languages natively. Alternatively, translate to English before encoding if an offline translation dataset is available.

9. **ROC AUC is the metric** — optimize BCE loss, report AUC at every epoch; do not threshold predictions in the submission.

10. **Competition type is Code** — the notebook IS the submission; it must reproduce results in ≤ 9 hours (Kaggle notebook time limit for code competitions).

---

## Prompt for AI Code Generator

Use the above specification to generate a complete, single-file Jupyter notebook
(`rsna_knee_solution.ipynb`) that:

- Runs end-to-end without errors in a Kaggle GPU environment
- Handles both DICOM MRI images and multilingual text reports
- Trains a multimodal deep learning model across 5 folds
- Evaluates each fold using macro ROC AUC
- Generates a valid `submission.csv` at `/kaggle/working/submission.csv`
- Uses only libraries from the recommended stack above (all installable in Kaggle)
- Includes defensive checks (assertions, try/except) so it does not crash on edge cases

Do NOT use:
- `wandb`, `neptune`, or any logging service that requires internet
- `tensorflow` (use PyTorch only)
- Hardcoded study IDs or filenames
- Any API key or external URL fetch

The notebook should be self-contained and production-quality.
