# ── 1. Dependency Verification ──────────────────────────────────────────────
import subprocess, sys

packages = {'timm': 'timm', 'albumentations': 'albumentations', 'pydicom': 'pydicom', 'scikit-learn': 'sklearn', 'scipy': 'scipy'}
for pip_name, mod_name in packages.items():
    try:
        __import__(mod_name)
    except ImportError:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pip_name], check=False)
        except Exception:
            pass
print('Dependency verification complete.', flush=True)


# --- CELL ---

# ── 2. Imports & Configuration ──────────────────────────────────────────────
import os, gc, glob, re, random, time, warnings, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import cv2, pydicom
from scipy.stats import rankdata

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TIMM_OFFLINE"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import GroupKFold

import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings('ignore')

SEED = 2026
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything()

KAGGLE_DIR = Path('/kaggle/input/rsna-knee-abnormality-detection')
BASE_DIR   = KAGGLE_DIR if KAGGLE_DIR.exists() else Path('.')

if os.name != 'nt' and Path('/kaggle/working').exists():
    OUTPUT_DIR = Path('/kaggle/working')
else:
    OUTPUT_DIR = Path('./v8_output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OFFICIAL_LABEL_COLS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    "Synovitis", "Baker's", 'Contusion', 'Fracture'
]

def get_safe_device():
    if torch.cuda.is_available():
        try:
            test_x = torch.zeros(1, 3, 224, 224, device='cuda')
            test_conv = nn.Conv2d(3, 16, 3).to('cuda')
            _ = test_conv(test_x)
            gpu_name = torch.cuda.get_device_name(0)
            print(f'CUDA Active: {gpu_name}', flush=True)
            return 'cuda', True
        except Exception as e:
            print(f'[CUDA fallback] {e}', flush=True)
            return 'cpu', False
    print('CUDA not available. Running on CPU.', flush=True)
    return 'cpu', False

DEVICE, USE_AMP = get_safe_device()

CFG = dict(
    img_size         = 336,
    crop_mm          = 130.0,
    n_slices         = 3 if DEVICE == 'cpu' else 16,
    slice_band       = (0.12, 0.88),
    batch_size       = 2 if DEVICE == 'cpu' else 4,
    grad_accum_steps = 2,
    n_epochs         = 1 if DEVICE == 'cpu' else 4,
    n_folds          = 5,
    n_folds_train    = 2,
    lr_backbone      = 1e-5,
    lr_head          = 2e-4,
    weight_decay     = 1e-2,
    device           = DEVICE,
    model_name       = 'resnet18' if DEVICE == 'cpu' else 'convnext_small',
    mixed_prec       = USE_AMP,
    num_workers      = 2 if DEVICE == 'cuda' else 0,
    text_dim         = 16,
)
print('Config:', CFG, flush=True)


# --- CELL ---

# ── 3. Multimodal Data Schema & Radiology Report Text Vectorizer ─────────
def find_file(filename: str):
    direct = BASE_DIR / filename
    if direct.exists():
        return str(direct)
    local_f = Path('.') / filename
    if local_f.exists():
        return str(local_f)
    matches = glob.glob(f'./**/{filename}', recursive=True)
    return matches[0] if matches else None

train_csv_path  = find_file('train.csv')
test_csv_path   = find_file('test.csv')
sub_csv_path    = find_file('sample_submission.csv')

if sub_csv_path and os.path.exists(sub_csv_path):
    sample_sub = pd.read_csv(sub_csv_path)
else:
    sample_sub = pd.DataFrame({
        'StudyInstanceUID': [f'test_study_{i}' for i in range(5)],
        **{c: [0.5]*5 for c in OFFICIAL_LABEL_COLS}
    })

ID_COL = 'StudyInstanceUID' if 'StudyInstanceUID' in sample_sub.columns else sample_sub.columns[0]
LABEL_COLS = [c for c in sample_sub.columns if c != ID_COL] or OFFICIAL_LABEL_COLS

if train_csv_path and os.path.exists(train_csv_path):
    train_df = pd.read_csv(train_csv_path)
else:
    train_df = pd.DataFrame({
        ID_COL: [f'train_study_{i}' for i in range(50)],
        'PatientSex': ['M' if i%2==0 else 'F' for i in range(50)],
        'ReportText': [f'Patient presents with knee swelling and tear in study {i}.' for i in range(50)],
        **{c: np.random.randint(0, 2, 50) for c in LABEL_COLS}
    })

if test_csv_path and os.path.exists(test_csv_path):
    test_df = pd.read_csv(test_csv_path)
else:
    test_df = pd.DataFrame({ID_COL: sample_sub[ID_COL].values})

# Extract or fit Radiology Report Text Embeddings
report_text_col = [c for c in train_df.columns if 'report' in c.lower() or 'text' in c.lower() or 'impression' in c.lower()]

if report_text_col:
    text_col = report_text_col[0]
    print(f'Found Radiology Report Text Column: {text_col}', flush=True)
    tfidf = TfidfVectorizer(max_features=CFG['text_dim'], stop_words='english')
    train_text_vecs = tfidf.fit_transform(train_df[text_col].fillna('')).toarray()
    if text_col in test_df.columns:
        test_text_vecs = tfidf.transform(test_df[text_col].fillna('')).toarray()
    else:
        test_text_vecs = np.zeros((len(test_df), CFG['text_dim']))
else:
    print('No text column found in CSV. Initializing default clinical feature vector.', flush=True)
    train_text_vecs = np.zeros((len(train_df), CFG['text_dim']))
    test_text_vecs  = np.zeros((len(test_df), CFG['text_dim']))

print(f'ID Column        : {ID_COL}', flush=True)
print(f'Target Labels    : ({len(LABEL_COLS)}) {LABEL_COLS[:4]}...', flush=True)
print(f'Train Shape      : {train_df.shape} | Test Shape: {test_df.shape}', flush=True)
print(f'Multimodal Text Feature Matrix Shape: {train_text_vecs.shape}', flush=True)


# --- CELL ---

# ── 4. 130mm FOV Bounding Cropper & Sequence Slot Classifier ────────────────
def extract_slice_num(filepath: str) -> int:
    fname = os.path.basename(filepath)
    nums = re.findall(r'\d+', fname)
    return int(nums[-1]) if nums else 0

def crop_130mm_knee_fov(dcm, img_size: int = CFG['img_size'], crop_mm: float = CFG['crop_mm']) -> np.ndarray:
    try:
        img = dcm.pixel_array.astype(np.float32)
        slope = float(getattr(dcm, 'RescaleSlope', 1.0) or 1.0)
        intercept = float(getattr(dcm, 'RescaleIntercept', 0.0) or 0.0)
        img = img * slope + intercept
        if getattr(dcm, 'PhotometricInterpretation', '') == 'MONOCHROME1':
            img = np.max(img) - img

        p1, p99 = np.percentile(img, [1, 99])
        if p99 > p1:
            img = np.clip((img - p1) / (p99 - p1) * 255.0, 0, 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)

        spacing = getattr(dcm, 'PixelSpacing', [1.0, 1.0])
        sp_y, sp_x = float(spacing[0]), float(spacing[1])
        h, w = img.shape

        crop_pixels_y = int(crop_mm / max(sp_y, 0.1))
        crop_pixels_x = int(crop_mm / max(sp_x, 0.1))

        crop_h = min(h, max(64, crop_pixels_y))
        crop_w = min(w, max(64, crop_pixels_x))

        start_y = max(0, (h - crop_h) // 2)
        start_x = max(0, (w - crop_w) // 2)

        cropped = img[start_y:start_y+crop_h, start_x:start_x+crop_w]
        return cv2.resize(cropped, (img_size, img_size))
    except Exception:
        return np.zeros((img_size, img_size), dtype=np.uint8)

STUDY_SLOT_MAP = {}
print('Indexing DICOM series & classifying FS/T1 sequence slots...', flush=True)
t0 = time.time()
input_dir = str(BASE_DIR)

if os.path.exists(input_dir):
    for root, _, files in os.walk(input_dir):
        dcms = [os.path.join(root, f) for f in files if f.lower().endswith('.dcm')]
        if not dcms:
            continue
        dcms.sort(key=extract_slice_num)
        
        folder_name = os.path.basename(root)
        parent_name = os.path.basename(os.path.dirname(root))
        folder_lower = (folder_name + " " + parent_name).lower()

        if 'sag' in folder_lower:
            plane = 'Sagittal'
        elif 'cor' in folder_lower:
            plane = 'Coronal'
        elif 'ax' in folder_lower:
            plane = 'Axial'
        else:
            plane = 'Sagittal'

        is_fs = any(fs_kw in folder_lower for fs_kw in ['fs', 'fat', 'fluid', 'pdfs', 't2', 'stir'])
        slot_key = f"{plane}_{'FS' if is_fs else 'NOFS'}"

        STUDY_SLOT_MAP.setdefault(folder_name, {}).setdefault(slot_key, []).extend(dcms)
        STUDY_SLOT_MAP.setdefault(parent_name, {}).setdefault(slot_key, []).extend(dcms)

print(f'Indexed {len(STUDY_SLOT_MAP)} study mappings in {time.time()-t0:.2f}s', flush=True)

def get_study_slot_dicoms(study_uid: str, plane: str = 'Sagittal', is_fs: bool = True) -> list:
    slots_map = STUDY_SLOT_MAP.get(str(study_uid), {})
    primary_slot = f"{plane}_{'FS' if is_fs else 'NOFS'}"
    fallback_slot = f"{plane}_{'NOFS' if is_fs else 'FS'}"

    if primary_slot in slots_map and slots_map[primary_slot]:
        return sorted(slots_map[primary_slot], key=extract_slice_num)
    if fallback_slot in slots_map and slots_map[fallback_slot]:
        return sorted(slots_map[fallback_slot], key=extract_slice_num)

    for k, files in slots_map.items():
        if plane in k and files:
            return sorted(files, key=extract_slice_num)
    return []


# --- CELL ---

# ── 5. Albumentations & Multimodal Dataset Class ───────────────────────────
def load_interior_volume(paths: list, n_slices: int = CFG['n_slices'], img_size: int = CFG['img_size']) -> np.ndarray:
    if not paths:
        return np.zeros((n_slices, img_size, img_size, 3), dtype=np.uint8)
    total = len(paths)
    
    b_low, b_high = CFG['slice_band']
    start_idx = int(total * b_low)
    end_idx   = int(total * b_high)
    if end_idx <= start_idx:
        start_idx, end_idx = 0, total - 1

    sub_paths = paths[start_idx:end_idx+1]
    n_sub = len(sub_paths)
    if n_sub <= n_slices:
        indices = list(range(n_sub)) + [n_sub - 1] * (n_slices - n_sub)
    else:
        indices = np.linspace(0, n_sub - 1, n_slices, dtype=int)

    stack = []
    for idx in indices:
        try:
            dcm = pydicom.dcmread(sub_paths[idx])
            s_img = crop_130mm_knee_fov(dcm, img_size=img_size)
        except Exception:
            s_img = np.zeros((img_size, img_size), dtype=np.uint8)
        s_rgb = np.stack([s_img, s_img, s_img], axis=-1)
        stack.append(s_rgb)
    return np.array(stack) # [N, H, W, 3]

def get_transforms(mode='train'):
    h = w = CFG['img_size']
    if mode == 'train':
        return A.Compose([
            A.RandomResizedCrop(size=(h, w), scale=(0.85, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(height=h, width=w),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

class MultimodalKneeMRIDatasetV8(Dataset):
    def __init__(self, df, text_vecs, split='train'):
        self.df = df.reset_index(drop=True)
        self.text_vecs = text_vecs
        self.split = split
        self.tfms = get_transforms(split)
        self.label_cols = LABEL_COLS if split != 'test' and all(c in df.columns for c in LABEL_COLS) else []

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = row[ID_COL] if ID_COL in row else row.iloc[0]

        sag_fs = get_study_slot_dicoms(uid, 'Sagittal', is_fs=True)
        cor_fs = get_study_slot_dicoms(uid, 'Coronal',  is_fs=True)
        ax_fs  = get_study_slot_dicoms(uid, 'Axial',    is_fs=True)

        sag_vol = load_interior_volume(sag_fs, CFG['n_slices'], CFG['img_size'])
        cor_vol = load_interior_volume(cor_fs, CFG['n_slices'], CFG['img_size'])
        ax_vol  = load_interior_volume(ax_fs,  CFG['n_slices'], CFG['img_size'])

        sag_t = torch.stack([self.tfms(image=sag_vol[i])['image'] for i in range(len(sag_vol))])
        cor_t = torch.stack([self.tfms(image=cor_vol[i])['image'] for i in range(len(cor_vol))])
        ax_t  = torch.stack([self.tfms(image=ax_vol[i])['image'] for i in range(len(ax_vol))])

        sex_val = str(row.get('PatientSex', 'M')).upper()
        sex_onehot = [1.0, 0.0] if sex_val == 'F' else [0.0, 1.0]

        # Combine Demographics + Diagnostic Text Embeddings into Tabular Vector
        t_feat = self.text_vecs[idx] if idx < len(self.text_vecs) else np.zeros(CFG['text_dim'])
        combined_tab = torch.tensor(np.concatenate([sex_onehot, t_feat]).astype(np.float32))

        item = {
            'sagittal': sag_t,
            'coronal':  cor_t,
            'axial':    ax_t,
            'tabular':  combined_tab,
            'study_uid': uid
        }
        if self.label_cols:
            item['labels'] = torch.tensor(row[self.label_cols].values.astype(np.float32))
        return item


# --- CELL ---

# ── 6. Multimodal MultiViewKneeNet Architecture ───────────────────────────
class SpatialAttentionPooling1D(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.GELU(),
            nn.Linear(in_features // 2, 1)
        )
    def forward(self, x): # [B, N_slices, C]
        w = torch.softmax(self.attn(x), dim=1)
        return (x * w).sum(dim=1) # [B, C]

class MultimodalKneeNetV8(nn.Module):
    def __init__(self, n_classes=len(LABEL_COLS), model_name=CFG['model_name'], text_dim=CFG['text_dim']):
        super().__init__()
        
        backbone_built = False
        candidates = [model_name, 'convnext_small', 'tf_efficientnet_b4', 'resnet34d', 'resnet18']
        for bname in candidates:
            try:
                self.backbone = timm.create_model(bname, pretrained=False, num_classes=0)
                print(f'Created MultimodalKneeNetV8 backbone: {bname} (pretrained=False)', flush=True)
                backbone_built = True
                break
            except Exception:
                continue
                
        if not backbone_built:
            self.backbone = timm.create_model('resnet18', pretrained=False, num_classes=0)

        in_feat = self.backbone.num_features
        self.sag_attn = SpatialAttentionPooling1D(in_feat)
        self.cor_attn = SpatialAttentionPooling1D(in_feat)
        self.ax_attn  = SpatialAttentionPooling1D(in_feat)

        fused_dim = in_feat * 3 + (2 + text_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(0.3),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, n_classes)
        )

    def _forward_plane(self, x, attn_pool):
        B, N, C, H, W = x.shape
        x_flat = x.view(B * N, C, H, W)
        feats = self.backbone(x_flat).view(B, N, -1)
        return attn_pool(feats)

    def forward(self, sag, cor, ax, tab):
        sag_pooled = self._forward_plane(sag, self.sag_attn)
        cor_pooled = self._forward_plane(cor, self.cor_attn)
        ax_pooled  = self._forward_plane(ax,  self.ax_attn)

        fused = torch.cat([sag_pooled, cor_pooled, ax_pooled, tab], dim=-1)
        return self.classifier(fused)

def build_knee_model_v8(n_classes=len(LABEL_COLS)):
    model = MultimodalKneeNetV8(n_classes=n_classes)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f'MultimodalKneeNetV8 built | Parameters: {trainable/1e6:.2f}M / {total/1e6:.2f}M', flush=True)
    return model


# --- CELL ---

# ── 7. Asymmetric Loss & Training Engine ──────────────────────────────────
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        mask = ~torch.isnan(y)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        x_m = x[mask]
        y_m = y[mask]

        targets = y_m
        anti_targets = 1 - targets

        xs_pos = torch.sigmoid(x_m)
        xs_neg = 1 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = anti_targets * torch.log(xs_neg.clamp(min=self.eps))

        if self.gamma_pos > 0:
            los_pos = los_pos * ((1 - xs_pos) ** self.gamma_pos)
        if self.gamma_neg > 0:
            los_neg = los_neg * (xs_pos ** self.gamma_neg)

        loss = - (los_pos + los_neg)
        return loss.mean()

try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1)

def train_one_epoch(model, loader, optimizer, scheduler, scaler, device):
    model.train()
    total_loss = 0.0
    use_amp = CFG['mixed_prec'] and str(device) == 'cuda'
    accum = CFG.get('grad_accum_steps', 2)
    optimizer.zero_grad()

    for idx, batch in enumerate(loader):
        sag = batch['sagittal'].to(device)
        cor = batch['coronal'].to(device)
        ax  = batch['axial'].to(device)
        tab = batch['tabular'].to(device)
        lbls = batch['labels'].to(device)

        with autocast(device_type='cuda' if str(device)=='cuda' else 'cpu', enabled=use_amp):
            logits = torch.nan_to_num(model(sag, cor, ax, tab), nan=0.0)
            loss   = criterion(logits, lbls) / accum

        if use_amp:
            scaler.scale(loss).backward()
            if (idx + 1) % accum == 0 or (idx + 1) == len(loader):
                has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
                if has_grads:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (idx + 1) % accum == 0 or (idx + 1) == len(loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * accum if not torch.isnan(loss) else 0.0

    if scheduler is not None:
        scheduler.step()

    return total_loss / max(1, len(loader))

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    use_amp = CFG['mixed_prec'] and str(device) == 'cuda'

    for batch in loader:
        sag = batch['sagittal'].to(device)
        cor = batch['coronal'].to(device)
        ax  = batch['axial'].to(device)
        tab = batch['tabular'].to(device)
        lbls = batch['labels'].to(device)

        with autocast(device_type='cuda' if str(device)=='cuda' else 'cpu', enabled=use_amp):
            logits = torch.nan_to_num(model(sag, cor, ax, tab), nan=0.0)
            loss   = criterion(logits, lbls)

        total_loss += loss.item() if not torch.isnan(loss) else 0.0
        preds = np.nan_to_num(torch.sigmoid(logits).cpu().numpy(), nan=0.5)
        all_preds.append(preds)
        all_labels.append(lbls.cpu().numpy())

    P = np.vstack(all_preds)  if all_preds  else np.zeros((0, len(LABEL_COLS)))
    L = np.vstack(all_labels) if all_labels else np.zeros((0, len(LABEL_COLS)))

    aucs, f1s = [], []
    for i in range(L.shape[1]):
        valid = ~np.isnan(L[:, i])
        if valid.sum() > 0 and len(np.unique(L[valid, i])) > 1:
            try:
                aucs.append(roc_auc_score(L[valid, i], P[valid, i]))
                f1s.append(f1_score(L[valid, i], (P[valid, i] > 0.5).astype(int)))
            except Exception:
                aucs.append(0.5); f1s.append(0.0)
        else:
            aucs.append(0.5); f1s.append(0.0)

    return total_loss / max(1, len(loader)), float(np.mean(aucs)), float(np.mean(f1s)), P


# --- CELL ---

# ── 8. Leakage-Free GroupKFold Training Loop ──────────────────────────────
def run_fold(fold, tr_df, vl_df, tr_vecs, vl_vecs):
    print(f'\n==================== FOLD {fold} ====================', flush=True)
    train_ds = MultimodalKneeMRIDatasetV8(tr_df, tr_vecs, split='train')
    val_ds   = MultimodalKneeMRIDatasetV8(vl_df, vl_vecs, split='val')

    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True, drop_last=False, num_workers=CFG['num_workers'])
    val_loader   = DataLoader(val_ds,   batch_size=CFG['batch_size']*2, shuffle=False, num_workers=CFG['num_workers'])

    model = build_knee_model_v8(n_classes=len(LABEL_COLS)).to(CFG['device'])

    backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n and p.requires_grad]
    head_params     = [p for n, p in model.named_parameters() if 'classifier' in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': CFG['lr_backbone']},
        {'params': head_params,     'lr': CFG['lr_head']},
    ], weight_decay=CFG['weight_decay'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['n_epochs'], eta_min=1e-6)
    scaler = GradScaler(enabled=CFG['mixed_prec'] and CFG['device'] == 'cuda')

    best_auc, best_preds, best_path = -1.0, None, str(OUTPUT_DIR / f'best_model_fold{fold}.pth')

    for epoch in range(1, CFG['n_epochs'] + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, CFG['device'])
        vl_loss, vl_auc, vl_f1, val_preds = validate(model, val_loader, CFG['device'])
        elapsed = time.time() - t0

        print(f'Epoch {epoch}/{CFG["n_epochs"]} [{elapsed:.1f}s] | Train Loss: {tr_loss:.4f} | Val Loss: {vl_loss:.4f} | Val AUC: {vl_auc:.4f} | Val F1: {vl_f1:.4f}', flush=True)

        if vl_auc > best_auc or best_preds is None:
            best_auc, best_preds = vl_auc, val_preds.copy()
            torch.save(model.state_dict(), best_path)
            print(f'  --> Saved checkpoint to {best_path} (Best Val AUC: {best_auc:.4f})', flush=True)

    del model; gc.collect()
    return best_preds, vl_df[ID_COL].values, best_auc

groups = train_df[ID_COL].values
gkf = GroupKFold(n_splits=CFG['n_folds'])
splits = list(gkf.split(train_df, groups=groups))

oof_preds = np.zeros((len(train_df), len(LABEL_COLS)))
fold_aucs = []

for fold, (tr_idx, vl_idx) in enumerate(splits):
    if fold >= CFG['n_folds_train']:
        break
    preds, study_ids, auc = run_fold(fold, train_df.iloc[tr_idx], train_df.iloc[vl_idx], train_text_vecs[tr_idx], train_text_vecs[vl_idx])
    oof_preds[vl_idx] = preds
    fold_aucs.append(auc)
    print(f'Fold {fold} Best Val AUC: {auc:.4f}', flush=True)

print(f'\nMean OOF AUC across {len(fold_aucs)} trained fold(s): {np.mean(fold_aucs):.4f}', flush=True)


# --- CELL ---

# ── 9. Percentile Rank-Averaged Official Submission Generator ────────────
def rank_columns_pct(arr: np.ndarray) -> np.ndarray:
    return pd.DataFrame(arr).rank(method='average', pct=True).to_numpy()

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_preds = []
    use_amp = CFG['mixed_prec'] and str(device) == 'cuda'

    for batch in loader:
        sag = batch['sagittal'].to(device)
        cor = batch['coronal'].to(device)
        ax  = batch['axial'].to(device)
        tab = batch['tabular'].to(device)

        with autocast(device_type='cuda' if str(device)=='cuda' else 'cpu', enabled=use_amp):
            logits = torch.nan_to_num(model(sag, cor, ax, tab), nan=0.0)
        preds = np.nan_to_num(torch.sigmoid(logits).cpu().numpy(), nan=0.5)
        all_preds.append(preds)

    return np.vstack(all_preds) if all_preds else np.zeros((0, len(LABEL_COLS)))

test_ds     = MultimodalKneeMRIDatasetV8(test_df, test_text_vecs, split='test')
test_loader = DataLoader(test_ds, batch_size=CFG['batch_size']*2, shuffle=False, drop_last=False, num_workers=CFG['num_workers'])

fold_preds_list = []
for fold in range(CFG['n_folds']):
    ckpt_path = str(OUTPUT_DIR / f'best_model_fold{fold}.pth')
    if not os.path.exists(ckpt_path):
        continue
    print(f'Loading checkpoint for percentile rank inference: {ckpt_path}', flush=True)
    model = build_knee_model_v8(n_classes=len(LABEL_COLS)).to(CFG['device'])
    model.load_state_dict(torch.load(ckpt_path, map_location=CFG['device']))
    p_fold = predict(model, test_loader, CFG['device'])
    
    p_ranked = rank_columns_pct(p_fold)
    fold_preds_list.append(p_ranked)
    del model; gc.collect()

if fold_preds_list:
    final_preds = np.mean(np.stack(fold_preds_list), axis=0)
    final_preds = np.clip(np.nan_to_num(final_preds, nan=0.5), 0.0, 1.0)
else:
    final_preds = np.full((len(test_df), len(LABEL_COLS)), 0.5)

sub_path = sub_csv_path if sub_csv_path and os.path.exists(sub_csv_path) else None
if sub_path:
    sub_df = pd.read_csv(sub_path)
else:
    sub_df = pd.DataFrame({
        'StudyInstanceUID': test_df[ID_COL].values if ID_COL in test_df.columns else ['test_0'],
        **{c: [0.5]*len(test_df) for c in OFFICIAL_LABEL_COLS}
    })

id_field    = 'StudyInstanceUID' if 'StudyInstanceUID' in sub_df.columns else sub_df.columns[0]
target_cols = [c for c in sub_df.columns if c != id_field]

if len(final_preds) == len(sub_df):
    for i, col in enumerate(target_cols):
        if i < final_preds.shape[1]:
            sub_df[col] = np.nan_to_num(final_preds[:, i], nan=0.5)

for col in target_cols:
    sub_df[col] = np.clip(sub_df[col].fillna(0.5), 0.0, 1.0)

out_file = OUTPUT_DIR / 'submission.csv'
sub_df.to_csv(out_file, index=False)
sub_df.to_csv('submission.csv', index=False)
print(f'Multimodal Submission generated at {out_file} (shape: {sub_df.shape})', flush=True)
print(sub_df.head(10), flush=True)
