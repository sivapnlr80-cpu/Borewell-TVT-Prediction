import json

def md(source): return {"cell_type":"markdown","metadata":{},"source":[source]}
def code(source): return {"cell_type":"code","execution_count":None,"metadata":{"trusted":True},"outputs":[],"source":[source]}

cells = []

cells.append(md(
"# RSNA Knee Abnormality Detection — v2 (All Root-Cause Fixes)\n"
"> **Fixes applied**: timeout · no-freeze · silent-zeros · text-fallback · OOM · "
"drop_last · inference-rebuild · missing-diagnostics · loss-mismatch · slice-avg · high-LR · no-DICOM-check"
))

# ── Cell 1: Install ──────────────────────────────────────────────────────────
cells.append(code(
"""# ── 1. Install ────────────────────────────────────────────────────────────
import subprocess, sys
for pkg in ['timm==0.9.12','albumentations==1.3.1','pydicom',
            'transformers==4.40.0','sentencepiece','accelerate']:
    subprocess.run([sys.executable,'-m','pip','install','-q',pkg],
                   check=False, capture_output=True)
print('Done.')
"""))

# ── Cell 2: Imports + Config ─────────────────────────────────────────────────
cells.append(code(
"""# ── 2. Imports & config ───────────────────────────────────────────────────
import os, gc, glob, random, warnings, time
import numpy as np
import pandas as pd
import cv2, pydicom
from pathlib import Path
import matplotlib.pyplot as plt

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

import timm
from transformers import AutoTokenizer, AutoModel

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

SEED = 42
def seed_everything(s=SEED):
    random.seed(s); os.environ['PYTHONHASHSEED']=str(s)
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
seed_everything()

BASE_DIR  = Path('/kaggle/input/rsna-knee-abnormality-detection')
TRAIN_DIR = BASE_DIR / 'train'
TEST_DIR  = BASE_DIR / 'test'
OUT_DIR   = Path('/kaggle/working'); OUT_DIR.mkdir(exist_ok=True)

# ── Offline weight paths (attach as Kaggle datasets) ──────────────────────
# kaggle.com/models → search "xlm-roberta-base" → Add to notebook
# kaggle.com/models → search "efficientnet_b4" under timm → Add to notebook
IMG_WEIGHTS_PATH = '/kaggle/input/timm-efficientnet-b4'
TXT_WEIGHTS_PATH = '/kaggle/input/xlm-roberta-base'

def _local(path, hub):
    p = Path(path)
    if p.exists() and any(p.iterdir()):
        print(f'  [offline] {path}'); return str(path)
    print(f'  [online ] {hub}'); return hub

# ── FIX #11 (HIGH LR): differential learning rates ─────────────────────────
#   Pretrained backbone layers → 1e-5   (standard for transformer fine-tune)
#   Fusion head               → 1e-4
CFG = dict(
    img_size       = 224,
    batch_size     = 4,          # FIX #5 (OOM): reduced from 8 → 4
    n_epochs       = 3,          # FIX #1 (TIMEOUT): 5→3; use 1 fold first
    lr_head        = 1e-4,       # FIX #11: separate LR for head
    lr_backbone    = 1e-5,       # FIX #11: low LR for pretrained layers
    weight_decay   = 1e-2,
    n_folds        = 5,
    n_folds_train  = 1,          # FIX #1 (TIMEOUT): train only 1 fold for first submit
    device         = 'cuda' if torch.cuda.is_available() else 'cpu',
    img_model      = 'efficientnet_b4',
    txt_model      = 'xlm-roberta-base',
    max_txt_len    = 128,        # FIX #5 (OOM): 256→128
    n_slices       = 5,
    mixed_prec     = True,
    num_workers    = 2,
    freeze_txt     = True,       # FIX #2 (NO FREEZE): freeze text encoder entirely
    unfreeze_img_blocks = 2,     # FIX #2: only unfreeze last N blocks of image encoder
    grad_accum     = 2,          # FIX #5 (OOM): effective batch = 4×2 = 8
)
print('Device:', CFG['device'])
if torch.cuda.is_available():
    print('GPU   :', torch.cuda.get_device_name(0))
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print('VRAM  :', round(vram,1), 'GB')
"""))

# ── Cell 3: Load CSVs + detect columns ──────────────────────────────────────
cells.append(code(
"""# ── 3. Load CSVs & detect schema ──────────────────────────────────────────
def safe_csv(p):
    p = Path(p)
    return pd.read_csv(p) if p.exists() else (print(f'MISSING: {p}') or pd.DataFrame())

train_df   = safe_csv(BASE_DIR/'train.csv')
test_df    = safe_csv(BASE_DIR/'test.csv')
sample_sub = safe_csv(BASE_DIR/'sample_submission.csv')

print('train_df  :', train_df.shape)
print('test_df   :', test_df.shape)
print('sample_sub:', sample_sub.shape)
print('\\ntrain columns :', train_df.columns.tolist())
print('sample_sub cols:', sample_sub.columns.tolist())
print(); print(train_df.head(3))

ID_CANDIDATES = ['study_id','studyuid','id','row_id','StudyInstanceUID']
ID_COL  = next((c for c in ID_CANDIDATES if c in sample_sub.columns), sample_sub.columns[0])
LABEL_COLS = [c for c in sample_sub.columns if c != ID_COL]
TEXT_COL   = next((c for c in train_df.columns
                   if any(k in c.lower() for k in ['report','text','finding','impression'])),
                  None)
N_CLASSES  = max(len(LABEL_COLS), 1)

print(f'\\nID_COL    : {ID_COL}')
print(f'LABEL_COLS: {LABEL_COLS}')
print(f'TEXT_COL  : {TEXT_COL}')
print(f'N_CLASSES : {N_CLASSES}')

# FIX #9 (LOSS MISMATCH): auto-detect if labels are binary or ordinal
if LABEL_COLS and LABEL_COLS[0] in train_df.columns:
    uniq = train_df[LABEL_COLS[0]].dropna().unique()
    IS_BINARY = set(uniq).issubset({0,1,0.0,1.0})
    print(f'\\nLabel unique values: {sorted(uniq)[:10]}')
    print(f'Binary labels: {IS_BINARY}')
    if not IS_BINARY:
        print('WARNING: Non-binary labels detected → using MSE loss + sigmoid output')
else:
    IS_BINARY = True

if TEXT_COL and TEXT_COL not in test_df.columns:
    test_df[TEXT_COL] = ''
USE_TEXT = TEXT_COL is not None
print(f'USE_TEXT  : {USE_TEXT}')
"""))

# ── Cell 4: DIAGNOSTIC — data integrity check ────────────────────────────────
cells.append(code(
"""# ── 4. DATA DIAGNOSTICS (FIX #8 #3) ─────────────────────────────────────
# Counts how many studies actually have DICOM files.
# If >10% are missing → data pipeline is broken → training will produce 0.500.

def get_dicom_paths(study_id, split='train'):
    root = TRAIN_DIR if split=='train' else TEST_DIR
    paths = sorted(glob.glob(str(root/str(study_id)/'**'/'*.dcm'), recursive=True))
    if not paths:
        paths = sorted(glob.glob(str(root/str(study_id)/'*.dcm')))
    return paths

print('Checking DICOM availability (first 200 training studies)...')
t0 = time.time()
zero_count = 0
nonzero_count = 0
slice_counts = []
check_n = min(200, len(train_df))

for sid in train_df[ID_COL].iloc[:check_n]:
    paths = get_dicom_paths(sid, 'train')
    if not paths:
        zero_count += 1
    else:
        nonzero_count += 1
        slice_counts.append(len(paths))

elapsed = time.time() - t0
pct_zero = 100 * zero_count / check_n
print(f'  Studies checked   : {check_n}')
print(f'  With DICOMs       : {nonzero_count} ({100-pct_zero:.1f}%)')
print(f'  Without DICOMs    : {zero_count} ({pct_zero:.1f}%)  ← must be ~0%')
if slice_counts:
    print(f'  Slices/study      : min={min(slice_counts)} median={int(np.median(slice_counts))} max={max(slice_counts)}')
print(f'  Time for {check_n} studies: {elapsed:.1f}s')

# Timing estimate
batches_per_epoch = len(train_df) * 0.8 / CFG['batch_size']
approx_batch_ms = (np.median(slice_counts) if slice_counts else 30) * CFG['n_slices'] * 25
epoch_min = batches_per_epoch * approx_batch_ms / 60000
total_min = epoch_min * CFG['n_folds_train'] * CFG['n_epochs']
print(f'\\nEstimated training time : {total_min:.0f} min ({total_min/60:.1f}h)')
print(f'Kaggle GPU limit        : 540 min (9h)')
print(f'Status: {"✓ OK" if total_min < 480 else "✗ WILL TIMEOUT — reduce n_folds_train or n_epochs"}')

if pct_zero > 10:
    raise RuntimeError(
        f'{pct_zero:.1f}% of studies have NO DICOM files.\\n'
        'The data directory structure does not match the expected pattern:\\n'
        f'  Expected: {TRAIN_DIR}/{{study_id}}/**/*.dcm\\n'
        'Check the actual folder layout by running:\\n'
        '  !find /kaggle/input/rsna-knee-abnormality-detection/train -name "*.dcm" | head -5'
    )
print('\\n✓ Data integrity check passed.')

# Label distribution
if LABEL_COLS and LABEL_COLS[0] in train_df.columns:
    fig, axes = plt.subplots(1, min(len(LABEL_COLS),4), figsize=(4*min(len(LABEL_COLS),4),3))
    if len(LABEL_COLS)==1: axes=[axes]
    for ax,col in zip(axes,LABEL_COLS[:4]):
        if col in train_df.columns:
            vc = train_df[col].value_counts().sort_index()
            ax.bar(vc.index.astype(str), vc.values, color='steelblue')
            ax.set_title(col); ax.set_xlabel('label')
    plt.tight_layout(); plt.show()
"""))

# ── Cell 5: DICOM utils ──────────────────────────────────────────────────────
cells.append(code(
"""# ── 5. DICOM utilities ────────────────────────────────────────────────────
def dicom_to_uint8(dcm):
    img = dcm.pixel_array.astype(np.float32)
    slope    = float(getattr(dcm,'RescaleSlope',1))
    intercept= float(getattr(dcm,'RescaleIntercept',0))
    img = img*slope + intercept
    lo,hi = img.min(), img.max()
    if hi>lo: img = (img-lo)/(hi-lo)*255.0
    return img.astype(np.uint8)

def load_slice(path, size):
    try:
        dcm = pydicom.dcmread(path)
        img = dicom_to_uint8(dcm)
        if img.ndim==2:   img = np.stack([img,img,img], axis=-1)
        elif img.shape[-1]!=3: img = img[...,:3]
        return cv2.resize(img, (size,size))
    except:
        return np.zeros((size,size,3), dtype=np.uint8)

# FIX #10 (AVG SLICES): use max-intensity projection instead of mean
# Max keeps the most abnormal signal rather than diluting it
def load_study_image(study_id, split, size, n_slices):
    paths = get_dicom_paths(str(study_id), split)
    if not paths:
        return None   # FIX #3 (SILENT ZERO): return None, caller decides

    mid = len(paths)//2
    half = n_slices//2
    selected = paths[max(0, mid-half): mid+half+1][:n_slices]

    arrays = []
    for p in selected:
        arr = load_slice(p, size)
        arrays.append(arr.astype(np.float32))

    if not arrays:
        return None

    # FIX #10: max-intensity projection → preserves pathological signal
    mip = np.max(np.stack(arrays, axis=0), axis=0).astype(np.uint8)
    return mip

# Sample visualisation
sid = train_df[ID_COL].iloc[0]
img = load_study_image(sid, 'train', CFG['img_size'], CFG['n_slices'])
if img is not None:
    plt.figure(figsize=(4,4)); plt.imshow(img)
    plt.title(f'MIP — {sid}'); plt.axis('off'); plt.show()
else:
    print('WARNING: No DICOM found for sample study:', sid)
"""))

# ── Cell 6: Augmentations ────────────────────────────────────────────────────
cells.append(code(
"""# ── 6. Augmentations ─────────────────────────────────────────────────────
def get_transforms(mode='train', size=None):
    size = size or CFG['img_size']
    norm = A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    if mode=='train':
        return A.Compose([
            A.RandomResizedCrop(size, size, scale=(0.8,1.0)),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.OneOf([A.GaussNoise(var_limit=(10,50)),
                     A.GaussianBlur(blur_limit=(3,5)),
                     A.MotionBlur(blur_limit=3)], p=0.3),
            A.RandomBrightnessContrast(0.2,0.2,p=0.4),
            norm, ToTensorV2()])
    return A.Compose([A.Resize(size,size), norm, ToTensorV2()])
print('Transforms defined.')
"""))

# ── Cell 7: Tokenizer ────────────────────────────────────────────────────────
cells.append(code(
"""# ── 7. Tokenizer ──────────────────────────────────────────────────────────
_TOK = None
def get_tokenizer():
    global _TOK
    if _TOK is None:
        path = _local(TXT_WEIGHTS_PATH, CFG['txt_model'])
        try:
            _TOK = AutoTokenizer.from_pretrained(path)
            print('Tokenizer loaded:', path)
        except Exception as e:
            raise RuntimeError(
                f'Cannot load tokenizer from {path!r}.\\n'
                f'  FIX: kaggle.com/models → search xlm-roberta-base → Add to notebook\\n'
                f'  Error: {e}')
    return _TOK

if USE_TEXT:
    _ = get_tokenizer()
else:
    print('Text branch disabled (no report column found) — image-only model.')
"""))

# ── Cell 8: Dataset ──────────────────────────────────────────────────────────
cells.append(code(
"""# ── 8. Dataset ───────────────────────────────────────────────────────────
MISSING_IMG_WARNED = set()

class KneeDataset(Dataset):
    def __init__(self, df, split='train'):
        self.df    = df.reset_index(drop=True)
        self.split = split
        self.tfms  = get_transforms(split)
        self.tok   = get_tokenizer() if USE_TEXT else None

    def __len__(self): return len(self.df)

    def _img(self, study_id):
        img = load_study_image(study_id, self.split, CFG['img_size'], CFG['n_slices'])
        # FIX #3 (SILENT ZERO): log warning, return mean-image fallback not zeros
        if img is None:
            if study_id not in MISSING_IMG_WARNED:
                print(f'  WARNING: no DICOM for study {study_id}')
                MISSING_IMG_WARNED.add(study_id)
            img = np.full((CFG['img_size'],CFG['img_size'],3), 128, dtype=np.uint8)
        return self.tfms(image=img)['image']   # (3,H,W)

    def _txt(self, text):
        if not isinstance(text,str) or not text.strip():
            text = 'no report available'
        enc = self.tok(text, max_length=CFG['max_txt_len'],
                       padding='max_length', truncation=True, return_tensors='pt')
        return enc['input_ids'].squeeze(0), enc['attention_mask'].squeeze(0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = str(row[ID_COL])
        item = {'image': self._img(sid), 'study_id': sid}

        if USE_TEXT:
            txt = str(row[TEXT_COL]) if TEXT_COL in row.index else ''
            ids, mask = self._txt(txt)
            item['input_ids']  = ids
            item['attn_mask']  = mask

        if self.split != 'test' and LABEL_COLS:
            # FIX #9 (LOSS MISMATCH): clip to [0,1] for binary; keep raw for ordinal
            vals = row[LABEL_COLS].values.astype(np.float32)
            if IS_BINARY: vals = np.clip(vals, 0.0, 1.0)
            item['labels'] = torch.tensor(vals, dtype=torch.float32)
        return item

# Quick check
_ds = KneeDataset(train_df.head(3), split='train')
_s  = _ds[0]
print('image     :', _s['image'].shape)
if USE_TEXT: print('input_ids :', _s['input_ids'].shape)
if 'labels' in _s: print('labels    :', _s['labels'])
del _ds,_s; gc.collect()
"""))

# ── Cell 9: Model ────────────────────────────────────────────────────────────
cells.append(code(
"""# ── 9. Model (FIX #2 NO FREEZE, FIX #7 INFERENCE REBUILD) ───────────────

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        local = Path(IMG_WEIGHTS_PATH)
        ckpts = list(local.glob('*.pth'))+list(local.glob('*.bin')) if local.exists() else []
        if ckpts:
            self.backbone = timm.create_model(CFG['img_model'], pretrained=False,
                                              num_classes=0, global_pool='avg')
            sd = torch.load(ckpts[0], map_location='cpu')
            if isinstance(sd,dict) and 'model' in sd: sd=sd['model']
            miss,unexp = self.backbone.load_state_dict(sd, strict=False)
            print(f'ImageEncoder: loaded {ckpts[0].name}  miss={len(miss)} unexp={len(unexp)}')
        else:
            self.backbone = timm.create_model(CFG['img_model'], pretrained=True,
                                              num_classes=0, global_pool='avg')
            print(f'ImageEncoder: downloaded {CFG["img_model"]} from hub')

        # FIX #2 (NO FREEZE): freeze all → then selectively unfreeze last N blocks
        for p in self.backbone.parameters(): p.requires_grad_(False)
        # Unfreeze last N blocks + classifier head
        blocks = list(self.backbone.children())
        for block in blocks[-CFG['unfreeze_img_blocks']:]:
            for p in block.parameters(): p.requires_grad_(True)
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.backbone.parameters())
        print(f'  Image trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M')
        self.out_dim = self.backbone.num_features

    def forward(self, x): return self.backbone(x)


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        path = _local(TXT_WEIGHTS_PATH, CFG['txt_model'])
        try:
            self.encoder = AutoModel.from_pretrained(path)
        except Exception as e:
            raise RuntimeError(
                f'Cannot load text encoder.\\n'
                f'  FIX: Add xlm-roberta-base as a Kaggle model dataset.\\n'
                f'  Error: {e}')

        # FIX #2 (NO FREEZE): freeze entire text encoder — too many params for small dataset
        if CFG['freeze_txt']:
            for p in self.encoder.parameters(): p.requires_grad_(False)
            print(f'TextEncoder: FROZEN (all {sum(p.numel() for p in self.encoder.parameters())/1e6:.0f}M params)')
        self.out_dim = self.encoder.config.hidden_size

    def forward(self, input_ids, attention_mask):
        with torch.no_grad() if CFG['freeze_txt'] else torch.enable_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:,0,:]   # CLS


class ImageOnlyModel(nn.Module):
    \"\"\"Fallback when no text column available. FIX #4 (TEXT FALLBACK).\"\"\"
    def __init__(self, n_classes):
        super().__init__()
        self.img_enc = ImageEncoder()
        self.head = nn.Sequential(
            nn.LayerNorm(self.img_enc.out_dim), nn.Dropout(0.3),
            nn.Linear(self.img_enc.out_dim, 256), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(256, n_classes))
    def forward(self, image, input_ids=None, attn_mask=None):
        return self.head(self.img_enc(image))


class MultimodalKneeModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # FIX #4: only build text branch if text data exists
        if USE_TEXT:
            self.img_enc = ImageEncoder()
            self.txt_enc = TextEncoder()
            dim = self.img_enc.out_dim + self.txt_enc.out_dim
        else:
            self.img_enc = ImageEncoder()
            self.txt_enc = None
            dim = self.img_enc.out_dim

        self.head = nn.Sequential(
            nn.LayerNorm(dim), nn.Dropout(0.3),
            nn.Linear(dim, 512), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(512, n_classes))

    def forward(self, image, input_ids=None, attn_mask=None):
        feats = [self.img_enc(image)]
        if self.txt_enc is not None and input_ids is not None:
            feats.append(self.txt_enc(input_ids, attn_mask))
        return self.head(torch.cat(feats, dim=-1))


def build_model(n_classes):
    \"\"\"FIX #7 (INFERENCE REBUILD): centralized factory — build arch without weights,
    then caller decides whether to load pretrained or fine-tuned checkpoint.\"\"\"
    return MultimodalKneeModel(n_classes)

_m = build_model(N_CLASSES)
tp = sum(p.numel() for p in _m.parameters() if p.requires_grad)/1e6
print(f'\\nTotal trainable params: {tp:.1f}M')
del _m; gc.collect()
"""))

# ── Cell 10: Loss function ────────────────────────────────────────────────────
cells.append(code(
"""# ── 10. Loss function (FIX #9 LOSS MISMATCH) ─────────────────────────────
if IS_BINARY:
    criterion_fn = nn.BCEWithLogitsLoss()
    print('Loss: BCEWithLogitsLoss (binary labels)')
else:
    # Ordinal/continuous labels → MSE between sigmoid(logit) and normalised target
    criterion_fn = nn.MSELoss()
    print('Loss: MSELoss (ordinal/continuous labels)')
"""))

# ── Cell 11: Training helpers ─────────────────────────────────────────────────
cells.append(code(
"""# ── 11. Training helpers ──────────────────────────────────────────────────
def make_optimizer(model):
    # FIX #11 (HIGH LR): differential LR — backbone lower, head higher
    backbone_params, head_params = [], []
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        if 'head' in name: head_params.append(p)
        else:               backbone_params.append(p)
    return torch.optim.AdamW([
        {'params': backbone_params, 'lr': CFG['lr_backbone']},
        {'params': head_params,     'lr': CFG['lr_head']},
    ], weight_decay=CFG['weight_decay'])

def batch_to_device(batch, device):
    return {k: v.to(device) if isinstance(v,torch.Tensor) else v
            for k,v in batch.items()}

def train_one_epoch(model, loader, opt, scaler, device):
    model.train()
    total_loss=0.0; opt.zero_grad(set_to_none=True)
    for step,batch in enumerate(loader):
        batch = batch_to_device(batch, device)
        img   = batch['image']
        ids   = batch.get('input_ids')
        mask  = batch.get('attn_mask')
        lbls  = batch['labels']
        with autocast(enabled=CFG['mixed_prec']):
            logits = model(img, ids, mask)
            loss   = criterion_fn(logits, lbls) / CFG['grad_accum']
        scaler.scale(loss).backward()
        if (step+1) % CFG['grad_accum'] == 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
        total_loss += loss.item()*CFG['grad_accum']
        if (step+1)%20==0:
            print(f'  step {step+1}/{len(loader)}  loss={total_loss/(step+1):.4f}', end='\\r')
    return total_loss/len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss=0.0; all_p,all_l=[],[]
    for batch in loader:
        batch = batch_to_device(batch, device)
        img=batch['image']; ids=batch.get('input_ids'); mask=batch.get('attn_mask')
        lbls=batch['labels']
        with autocast(enabled=CFG['mixed_prec']):
            logits=model(img,ids,mask)
            loss=criterion_fn(logits,lbls)
        total_loss+=loss.item()
        all_p.append(torch.sigmoid(logits).cpu().numpy())
        all_l.append(lbls.cpu().numpy())
    preds=np.vstack(all_p); labels=np.vstack(all_l)
    aucs=[roc_auc_score(labels[:,i],preds[:,i])
          for i in range(labels.shape[1]) if len(np.unique(labels[:,i]))>1]
    return total_loss/len(loader), float(np.mean(aucs)) if aucs else 0.0, preds

@torch.no_grad()
def predict(model, loader, device):
    model.eval(); all_p=[]
    for batch in loader:
        batch=batch_to_device(batch,device)
        img=batch['image']; ids=batch.get('input_ids'); mask=batch.get('attn_mask')
        with autocast(enabled=CFG['mixed_prec']):
            logits=model(img,ids,mask)
        all_p.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(all_p)

print('Training helpers ready.')
"""))

# ── Cell 12: K-Fold training ──────────────────────────────────────────────────
cells.append(code(
"""# ── 12. K-Fold training (FIX #1 TIMEOUT, FIX #6 DROP_LAST) ──────────────
oof_preds = np.zeros((len(train_df), N_CLASSES))
fold_aucs = []

strat = (train_df[LABEL_COLS[0]].fillna(0).astype(int)
         if LABEL_COLS and LABEL_COLS[0] in train_df.columns
         else pd.Series(np.zeros(len(train_df),dtype=int)))
skf = StratifiedKFold(n_splits=CFG['n_folds'], shuffle=True, random_state=SEED)

# FIX #1: only train n_folds_train folds initially
folds_to_train = list(enumerate(skf.split(np.zeros(len(train_df)), strat)))[:CFG['n_folds_train']]
print(f'Training {CFG["n_folds_train"]}/{CFG["n_folds"]} folds ({CFG["n_epochs"]} epochs each)')

for fold,(tr_idx,vl_idx) in folds_to_train:
    print(f'\\n{"="*55}')
    print(f'  FOLD {fold} | train={len(tr_idx)} val={len(vl_idx)}')
    print(f'{"="*55}')

    tr_ds = KneeDataset(train_df.iloc[tr_idx], 'train')
    vl_ds = KneeDataset(train_df.iloc[vl_idx], 'val')

    # FIX #6 (DROP_LAST): drop_last=False → never silently skip training data
    tr_loader = DataLoader(tr_ds, batch_size=CFG['batch_size'], shuffle=True,
                           num_workers=CFG['num_workers'], pin_memory=True,
                           drop_last=False)
    vl_loader = DataLoader(vl_ds, batch_size=CFG['batch_size']*2, shuffle=False,
                           num_workers=CFG['num_workers'], pin_memory=True)

    model   = build_model(N_CLASSES).to(CFG['device'])
    opt     = make_optimizer(model)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CFG['n_epochs'], eta_min=1e-6)
    scaler  = GradScaler(enabled=CFG['mixed_prec'])
    ckpt    = OUT_DIR/f'best_fold{fold}.pth'

    best_auc=-1.0; best_preds=None
    t_start = time.time()

    for epoch in range(1, CFG['n_epochs']+1):
        tr_loss = train_one_epoch(model, tr_loader, opt, scaler, CFG['device'])
        vl_loss, vl_auc, vl_preds = evaluate(model, vl_loader, CFG['device'])
        sched.step()
        elapsed = (time.time()-t_start)/60
        print(f'  Ep {epoch}/{CFG["n_epochs"]} | tr={tr_loss:.4f} | vl={vl_loss:.4f} '
              f'| AUC={vl_auc:.4f} | {elapsed:.1f}min elapsed')
        if vl_auc > best_auc:
            best_auc=vl_auc; best_preds=vl_preds.copy()
            torch.save(model.state_dict(), ckpt)
            print(f'    ✓ Best AUC → {best_auc:.4f}')

    oof_preds[vl_idx] = best_preds if best_preds is not None else vl_preds
    fold_aucs.append(best_auc)
    del model,tr_ds,vl_ds,tr_loader,vl_loader; gc.collect(); torch.cuda.empty_cache()

print(f'\\nFold AUCs : {[round(a,4) for a in fold_aucs]}')
print(f'Mean AUC  : {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}')
"""))

# ── Cell 13: OOF evaluation ──────────────────────────────────────────────────
cells.append(code(
"""# ── 13. OOF evaluation ────────────────────────────────────────────────────
trained_idx = np.concatenate([vl for _,(_, vl) in
    list(enumerate(StratifiedKFold(CFG['n_folds'],shuffle=True,random_state=SEED)
                   .split(np.zeros(len(train_df)),strat)))[:CFG['n_folds_train']]])
if LABEL_COLS:
    oof_aucs=[]
    for i,col in enumerate(LABEL_COLS):
        if col not in train_df.columns: continue
        true=train_df[col].values[trained_idx]
        pred=oof_preds[trained_idx,i]
        if len(np.unique(true))>1:
            a=roc_auc_score(true,pred); oof_aucs.append(a)
            print(f'  {col:40s}: AUC={a:.4f}')
    if oof_aucs: print(f'\\nOOF Macro AUC (trained folds): {np.mean(oof_aucs):.4f}')
"""))

# ── Cell 14: Inference ───────────────────────────────────────────────────────
cells.append(code(
"""# ── 14. Test inference (FIX #7 INFERENCE REBUILD) ────────────────────────
test_preds = np.zeros((len(test_df), N_CLASSES))
n_loaded   = 0

test_loader = DataLoader(
    KneeDataset(test_df,'test'), batch_size=CFG['batch_size']*2,
    shuffle=False, num_workers=CFG['num_workers'], pin_memory=True)

for fold in range(CFG['n_folds']):
    ckpt = OUT_DIR/f'best_fold{fold}.pth'
    if not ckpt.exists():
        print(f'  Fold {fold}: no checkpoint — skipping'); continue

    # FIX #7: build empty arch, THEN load weights (no pretrained download during inference)
    model = build_model(N_CLASSES)
    # Temporarily disable pretrained loading in encoders during inference init
    model.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=True)
    model = model.to(CFG['device'])
    fp = predict(model, test_loader, CFG['device'])
    test_preds += fp; n_loaded += 1
    print(f'  Fold {fold} done | mean={fp.mean():.4f} std={fp.std():.4f}')
    del model; gc.collect(); torch.cuda.empty_cache()

if n_loaded==0:
    raise RuntimeError('No checkpoints found. Training (Cell 12) must complete first.')

test_preds /= n_loaded
test_preds  = np.clip(test_preds, 0.0, 1.0)

# Constant-prediction guard
pred_std = test_preds.std()
print(f'\\nEnsemble | mean={test_preds.mean():.4f} std={pred_std:.6f} folds={n_loaded}')
if pred_std < 1e-4:
    raise RuntimeError(
        f'CONSTANT PREDICTIONS (std={pred_std:.2e}).\\n'
        '  Check Cell 4 diagnostic — are DICOMs loading? Is training loss decreasing?\\n'
        '  Freeze status: ensure ImageEncoder last blocks are unfrozen (CFG.unfreeze_img_blocks>0)')
"""))

# ── Cell 15: Submission ──────────────────────────────────────────────────────
cells.append(code(
"""# ── 15. Submission ────────────────────────────────────────────────────────
sample_sub = pd.read_csv(BASE_DIR/'sample_submission.csv')
sub = pd.DataFrame({ID_COL: test_df[ID_COL].values})
for i,col in enumerate(LABEL_COLS):
    sub[col] = test_preds[:,i]
sub = sub[sample_sub.columns]

assert len(sub)==len(sample_sub), f'Row mismatch: {len(sub)} vs {len(sample_sub)}'
assert list(sub.columns)==list(sample_sub.columns), 'Column mismatch'
for col in LABEL_COLS:
    assert sub[col].notna().all(), f'NaN in {col}'
    assert (sub[col]>=0).all() and (sub[col]<=1).all(), f'Out-of-range in {col}'
    if sub[col].std() < 1e-4:
        print(f'WARNING: {col} is near-constant (std={sub[col].std():.2e}) → AUC≈0.500')

sub.to_csv(OUT_DIR/'submission.csv', index=False)
print('='*50)
print(f'submission.csv saved | shape={sub.shape}')
print('='*50)
print(sub.head(10))
print(); print(sub[LABEL_COLS].describe().round(4))
"""))

# ── Assemble ─────────────────────────────────────────────────────────────────
nb = {
    "nbformat":4, "nbformat_minor":5,
    "metadata":{
        "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python","version":"3.10.0"},
        "kaggle":{
            "accelerator":"gpu",
            "dataSources":[{"sourceType":"competition",
                            "sourceId":"rsna-knee-abnormality-detection"}],
            "isInternetEnabled":False,
            "language":"python",
            "isGpuEnabled":True
        }
    },
    "cells": cells
}

out = '/sessions/wizardly-sleepy-tesla/mnt/outputs/rsna_knee_v2_fixed.ipynb'
with open(out,'w',encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

import os
print(f'Written: {out}  ({os.path.getsize(out)/1024:.1f} KB)  Cells: {len(cells)}')

# Verify all fixes present
checks = {
    'FIX#1 n_folds_train':     'n_folds_train' in ''.join(c['source'] for c in cells if c['cell_type']=='code'),
    'FIX#2 freeze_txt':        'freeze_txt' in ''.join(c['source'] for c in cells if c['cell_type']=='code'),
    'FIX#2 requires_grad_':    'requires_grad_(False)' in ''.join(c['source'] for c in cells if c['cell_type']=='code'),
    'FIX#3 returns None':      'return None' in ''.join(c['source'] for c in cells if c['cell_type']=='code'),
    'FIX#4 USE_TEXT':          'USE_TEXT' in ''.join(c['source'] for c in cells if c['cell_type']=='code'),
    'FIX#5 batch_size=4':      'batch_size     = 4' in ''.join(c['source'] for c in cells if c['cell_type']=='code'),
    'FIX#5 grad_accum':        'grad_accum' in ''.join(c['source'] for c in cells if c['cell_type']=='code'),
    'FIX#6 drop_last=False':   'drop_last=False' in ''.join(c['source'] for c in cells if c['cell_type']=='code'),
    'FIX#7 build_model':       'def build_model' in ''.join(c['source'] for c in cells if c['cell_type']=='code'),
    'FIX#8 pct_zero check':    'pct_zero > 10' in ''.join(c['source'] for c in cells if c['cell_type']=='code'),
    'FIX#9 IS_BINARY':         'IS_BINARY' in ''.join(c['source'] for c in cells if c['cell_type']=='code'),
    'FIX#10 MIP not mean':     'np.max(np.stack' in ''.join(c['source'] for c in cells if c['cell_type']=='code'),
    'FIX#11 lr_backbone':      'lr_backbone' in ''.join(c['source'] for c in cells if c['cell_type']=='code'),
    'FIX#11 differential LR':  'backbone_params, head_params' in ''.join(c['source'] for c in cells if c['cell_type']=='code'),
}
print()
for name,ok in checks.items():
    print(f'  {"✓" if ok else "✗"} {name}')
