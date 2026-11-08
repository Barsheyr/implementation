#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Explainable multimodal deep learning for cassava disease detection and severity estimation.
Implementation of the methodology of Chapter 3 (MSc dissertation).

Variants
    M0  image-only baseline (EfficientNet-B0 branch only)
    M1  proposed multimodal model (image branch + environmental MLP, feature-level fusion)
    M2  shuffled-environment control (as M1, but environmental vectors permuted among images)

Typical use on Kaggle (T4 GPU, competition data attached, Internet on):
    python cassava_experiment.py --make-grading-sheet          # optional, once
    python cassava_experiment.py                               # full run (about 2 h on a T4)
    python cassava_experiment.py --quick                       # one seed, no tuning (about 25 min)

Nothing in this file invents data. If the weather request fails the run stops.
A --smoke run uses a tiny synthetic fixture to test the code and is stamped
"SMOKE TEST - NOT REAL RESULTS" on every output.
"""
import argparse
import copy
import datetime as dt
import glob
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================ 0. settings
CLASS_NAMES = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
CLASS_FULL = {"CBB": "Cassava Bacterial Blight", "CBSD": "Cassava Brown Streak Disease",
              "CGM": "Cassava Green Mottle", "CMD": "Cassava Mosaic Disease", "Healthy": "Healthy"}
K = 5
ENV_NAMES = ["T2M", "RH2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"]
ENV_LABELS = {"T2M": "Mean temperature", "RH2M": "Mean relative humidity",
              "PRECTOTCORR": "Total rainfall", "ALLSKY_SFC_SW_DWN": "Mean solar irradiance"}
MINORITY = [0, 1, 2, 4]          # every class except CMD gets the stronger augmentation
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLIT_SEED = 2026
SMOKE_BANNER = "SMOKE TEST - NOT REAL RESULTS"

API = "https://power.larc.nasa.gov/api/temporal/daily/point"
ZONES = {"Northern": ("Gulu", 2.78, 32.30), "Eastern": ("Soroti", 1.71, 33.61),
         "Central": ("Masaka", -0.34, 31.73), "Western": ("Mbarara", -0.61, 30.66)}
PERIOD = ("20180101", "20191231")
SEASONS = {"MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11), "DJF": (12, 1, 2)}

DEFAULTS = dict(
    img_size=224, batch_size=32, fusion_w=128, img_repr=256, env_repr=64, lam=1.0,
    lr_stage1=1e-3, lr_stage2=1e-4, epochs_stage1=3, epochs_stage2=8, patience=3,
    seeds=[42, 7, 2024], unfreeze_blocks=3, rotation_deg=30, jitter_standard=0.2, jitter_strong=0.4,
    n_bootstrap=2000, shap_background=100, shap_samples=200,
    grade_per_class={"train": 25, "val": 10, "test": 15},
    tune_grid={"lr_stage1": [1e-3, 3e-4], "fusion_w": [128, 256]}, tune_epochs=(2, 3),
)

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
PAL = {"M0": "#0072B2", "M1": "#D55E00", "M2": "#009E73"}


class Logger:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        line = f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def banner(fig, smoke):
    if smoke:
        fig.text(0.5, 0.995, SMOKE_BANNER, ha="center", va="top", color="red", fontsize=8, weight="bold")


def save_fig(fig, path, smoke):
    banner(fig, smoke)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================================ 1. image data
def _load_one(args):
    path, size = args
    try:
        with open(path, "rb") as f:
            raw = f.read()
        md5 = hashlib.md5(raw).hexdigest()
        im = Image.open(path)
        im.draft("RGB", (size * 2, size * 2))
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8), md5
    except Exception:
        return None, None


def find_data_dir(requested=None, log=print, root="/kaggle/input"):
    """Return the folder that holds train.csv and train_images/. Searches /kaggle/input when --data is not given."""
    def ok(d):
        return os.path.isfile(os.path.join(d, "train.csv")) and os.path.isdir(os.path.join(d, "train_images"))
    if requested:
        if ok(requested):
            return requested
        sys.exit(f"--data {requested} does not contain train.csv and train_images/.")
    for sub in ("competitions/cassava-leaf-disease-classification", "cassava-leaf-disease-classification"):
        if ok(os.path.join(root, sub)):
            log("data folder found:", os.path.join(root, sub))
            return os.path.join(root, sub)
    if os.path.isdir(root):
        for dp, dn, fn in os.walk(root, followlinks=True):
            if dp[len(root):].count(os.sep) > 4:
                dn[:] = []
                continue
            if "train.csv" in fn and "train_images" in dn:
                log("data folder found:", dp)
                return dp
        listing = []
        for dp, dn, fn in os.walk(root, followlinks=True):
            if dp[len(root):].count(os.sep) >= 2:
                dn[:] = []
            listing.append(dp)
        sys.exit("Could not find train.csv with a train_images folder under /kaggle/input.\n"
                 "Folders present: " + "; ".join(listing[:20]) + "\n"
                 "Fix: right-hand panel > Add Input > search 'Cassava Leaf Disease Classification' > Add "
                 "(you must have joined the competition with the same account), then run the cell again. "
                 "Or pass --data <folder containing train.csv>.")
    sys.exit("No /kaggle/input folder. Pass --data <folder containing train.csv and train_images/>.")


def prepare_images(data_dir, out, cfg, log, limit=None):
    """Read train.csv, drop unreadable files and exact duplicates, cache 224x224 uint8 array."""
    cache = os.environ.get("CASSAVA_CACHE") or ("/kaggle/temp/cassava_cache" if os.path.isdir("/kaggle/temp")
                                                 else os.path.join(out, "cache"))
    os.makedirs(cache, exist_ok=True)
    idx_csv, arr_path = os.path.join(cache, "index.csv"), os.path.join(cache, f"images_{cfg['img_size']}.npy")
    summ_path = os.path.join(cache, "load_summary.json")
    if os.path.exists(idx_csv) and os.path.exists(arr_path) and os.path.exists(summ_path):
        log("using cached images from", cache)
        sm = json.load(open(summ_path))
        sm["cache_dir"] = cache
        return pd.read_csv(idx_csv), np.load(arr_path, mmap_mode="r"), sm
    df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=0).reset_index(drop=True)
    n_listed = len(df)
    paths = [os.path.join(data_dir, "train_images", i) for i in df["image_id"]]
    size = cfg["img_size"]
    log(f"reading {n_listed} images and resizing to {size}x{size} ...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(2, (os.cpu_count() or 2))) as ex:
        res = list(ex.map(_load_one, [(p, size) for p in paths], chunksize=64))
    seen, keep, imgs, n_bad, n_dup = set(), [], [], 0, 0
    for i, (arr, md5) in enumerate(res):
        if arr is None:
            n_bad += 1
            continue
        if md5 in seen:
            n_dup += 1
            continue
        seen.add(md5)
        keep.append(i)
        imgs.append(arr)
    df = df.iloc[keep].reset_index(drop=True)
    arr = np.stack(imgs)
    del imgs, res
    np.save(arr_path, arr)
    df.to_csv(idx_csv, index=False)
    summ = {"n_listed": n_listed, "n_unreadable": n_bad, "n_duplicates_removed": n_dup, "n_final": len(df)}
    json.dump(summ, open(summ_path, "w"))
    summ["cache_dir"] = cache
    log(f"cached {len(df)} images in {time.time() - t0:.0f}s; unreadable={n_bad}, exact duplicates removed={n_dup}")
    return df, np.load(arr_path, mmap_mode="r"), summ


def stratified_split(labels, seed=SPLIT_SEED):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(labels))
    tr, rest = train_test_split(idx, test_size=0.2, stratify=labels, random_state=seed)
    va, te = train_test_split(rest, test_size=0.5, stratify=labels[rest], random_state=seed)
    part = np.empty(len(labels), dtype=object)
    part[tr], part[va], part[te] = "train", "val", "test"
    return part


# ============================================================================ 2. environmental data
def _mock_zone_json(lat, lon, period):
    """Deterministic fake API-shaped response. ONLY used by --smoke --mock-weather."""
    rng = np.random.default_rng(int(abs(lat * 1000 + lon * 10)))
    days = pd.date_range(period[0], period[1], freq="D")
    par = {}
    for p, (m, s) in zip(ENV_NAMES, [(22, 2), (70, 10), (3, 4), (5.5, 1)]):
        v = np.clip(rng.normal(m, s, len(days)), 0, None)
        v[rng.random(len(days)) < 0.01] = -999
        par[p] = {d.strftime("%Y%m%d"): float(x) for d, x in zip(days, v)}
    return {"properties": {"parameter": par}}


def fetch_zone(lat, lon, period, mock=False, retries=4):
    if mock:
        return _mock_zone_json(lat, lon, period), "MOCK (smoke test only)"
    import requests
    q = {"parameters": ",".join(ENV_NAMES), "community": "AG", "latitude": lat, "longitude": lon,
         "start": period[0], "end": period[1], "format": "JSON"}
    for k in range(retries):
        try:
            r = requests.get(API, params=q, timeout=120)
            if r.status_code == 200:
                return r.json(), r.url
            print(f"NASA POWER answered HTTP {r.status_code} for ({lat},{lon}), attempt {k + 1}: {r.text[:200]}", flush=True)
        except Exception as e:
            print(f"NASA POWER request error for ({lat},{lon}), attempt {k + 1}: {e!r}", flush=True)
        time.sleep(5 * (k + 1))
    raise SystemExit(f"NASA POWER request failed for ({lat},{lon}). Switch Internet on, or run fetch_weather.py "
                     "elsewhere and pass --weather-dir. No weather data are invented.")


def build_profiles(period=PERIOD, zones=ZONES, mock=False):
    rows = []
    meta = {"zones": {}, "period": list(period), "api": API, "parameters": ENV_NAMES, "mock": bool(mock),
            "fetch_time_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "Z", "missing_value_code": -999,
            "seasons": {k: list(v) for k, v in SEASONS.items()}}
    for zname, (town, lat, lon) in zones.items():
        js, url = fetch_zone(lat, lon, period, mock)
        par = js["properties"]["parameter"]
        df = pd.DataFrame({p: pd.Series(par[p]) for p in ENV_NAMES})
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        n_missing = int((df == -999).any(axis=1).sum())
        df = df.replace(-999, np.nan)
        meta["zones"][zname] = {"town": town, "latitude": lat, "longitude": lon, "request_url": url,
                                "n_days": int(len(df)), "n_days_with_missing_value": n_missing}
        for sname, months in SEASONS.items():
            d = df[df.index.month.isin(months)]
            sy = d.index.year + (d.index.month == 12).astype(int)
            full = d.groupby(sy).size()
            keep = full[full >= 0.9 * (92 if sname in ("JJA", "SON") else 90)].index
            msk = np.isin(sy, keep)
            d, sy = d[msk], sy[msk]
            if len(d) == 0:
                raise SystemExit(f"No complete {sname} season for {zname} in the period")
            rain = d["PRECTOTCORR"].groupby(sy).sum(min_count=1)
            rows.append({"profile_id": len(rows), "zone": zname, "season": sname,
                         "T2M": d["T2M"].mean(), "RH2M": d["RH2M"].mean(), "PRECTOTCORR": rain.mean(),
                         "ALLSKY_SFC_SW_DWN": d["ALLSKY_SFC_SW_DWN"].mean(), "n_seasons_used": int(len(keep))})
    return pd.DataFrame(rows), meta


def get_profiles(out, weather_dir, mock, log):
    if weather_dir:
        prof = pd.read_csv(os.path.join(weather_dir, "weather_profiles.csv"))
        meta = json.load(open(os.path.join(weather_dir, "weather_meta.json")))
        log("loaded weather profiles from", weather_dir)
    else:
        log("requesting NASA POWER daily data for", list(ZONES), "...")
        prof, meta = build_profiles(mock=mock)
    prof.to_csv(os.path.join(out, "weather_profiles.csv"), index=False)
    json.dump(meta, open(os.path.join(out, "weather_meta.json"), "w"), indent=2)
    return prof, meta


# ============================================================================ 3. severity grading
def make_grading_sheet(df, part, out, cfg, log, smoke):
    """Write a stratified sample of diseased images for visual grading (Healthy images are grade 1 by rule)."""
    d = os.path.join(out, "severity_grading")
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(SPLIT_SEED + 1)
    rows = []
    for p in ["train", "val", "test"]:
        for c in range(4):
            ids = np.where((part == p) & (df["label"].values == c))[0]
            take = rng.choice(ids, size=min(cfg["grade_per_class"][p], len(ids)), replace=False)
            rows += [(int(i), p, CLASS_NAMES[c]) for i in take]
    rng.shuffle(rows)
    sheet = pd.DataFrame([{"code": f"{k + 1:03d}", "image_id": df["image_id"][i], "partition": p, "disease": dis,
                           "grade": ""} for k, (i, p, dis) in enumerate(rows)])
    sheet.to_csv(os.path.join(d, "grading_sheet.csv"), index=False)
    return sheet, d


def draw_montages(sheet, data_dir, d, smoke):
    n_per, cols = 12, 4
    for s in range(0, len(sheet), n_per):
        chunk = sheet.iloc[s:s + n_per]
        rows = math.ceil(len(chunk) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.0 + 0.8))
        axes = np.atleast_2d(axes)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (_, r) in zip(axes.ravel(), chunk.iterrows()):
            try:
                ax.imshow(Image.open(os.path.join(data_dir, "train_images", r["image_id"])).convert("RGB"))
            except Exception:
                pass
            ax.set_title(f"{r['code']}   ({r['disease']})", fontsize=12, weight="bold")
        fig.suptitle("Grade each leaf 1-5:  1 = no symptoms   2 = mild   3 = moderate   4 = severe   5 = very severe",
                     fontsize=11, y=0.995)
        save_fig(fig, os.path.join(d, f"montage_{s // n_per + 1:02d}.jpg"), smoke)


def load_grades(out, df, part, cfg, grades_text, log):
    """Return (grade array, mask array, summary). Grades come from severity_grades.csv or pasted text."""
    n = len(df)
    grade, mask = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
    d = os.path.join(out, "severity_grading")
    sheet_path = os.path.join(d, "grading_sheet.csv")
    gfile = os.path.join(d, "severity_grades.csv")
    if grades_text and grades_text.strip() and os.path.exists(sheet_path):
        rows = [ln.split() for ln in grades_text.strip().splitlines() if ln.strip() and not ln.startswith("#")]
        pd.DataFrame(rows, columns=["code", "grade"]).to_csv(gfile + ".pasted", index=False)
        sheet = pd.read_csv(sheet_path, dtype=str)
        m = dict(rows)
        sheet["grade"] = sheet["code"].map(m)
        sheet[["image_id", "grade"]].dropna().to_csv(gfile, index=False)
    if not os.path.exists(gfile):
        log("no severity grades found: the severity head will not be trained and severity metrics are not reported")
        return grade, mask, {"available": False, "n_graded_by_partition": {"train": 0, "val": 0, "test": 0},
                             "grade_counts": {}}
    g = pd.read_csv(gfile)
    g["grade"] = pd.to_numeric(g["grade"], errors="coerce")
    g = g.dropna()
    g = g[(g["grade"] >= 1) & (g["grade"] <= 5)]
    pos = {k: i for i, k in enumerate(df["image_id"])}
    for iid, gr in zip(g["image_id"], g["grade"]):
        if iid in pos:
            grade[pos[iid]], mask[pos[iid]] = gr, 1.0
    # stratified Healthy sample of the same per-partition size gets grade 1 by rule
    rng = np.random.default_rng(SPLIT_SEED + 2)
    for p in ["train", "val", "test"]:
        ids = np.where((part == p) & (df["label"].values == 4))[0]
        take = rng.choice(ids, size=min(cfg["grade_per_class"][p], len(ids)), replace=False)
        grade[take], mask[take] = 1.0, 1.0
    summ = {"available": True,
            "n_graded_by_partition": {p: int(mask[part == p].sum()) for p in ["train", "val", "test"]},
            "grade_counts": {str(k): int(((grade == k) & (mask == 1)).sum()) for k in range(1, 6)}}
    log("severity grades loaded:", summ["n_graded_by_partition"])
    return grade, mask, summ


# ============================================================================ 4. model
class Net(nn.Module):
    """EfficientNet-B0 image branch, optional environmental MLP, feature-level fusion, two heads."""

    def __init__(self, variant="M1", fusion_w=128, img_repr=256, env_repr=64, pretrained=True):
        super().__init__()
        self.variant = variant
        self.frozen_bn = pretrained       # BN statistics stay pretrained; a from-scratch smoke model trains them
        w = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = torchvision.models.efficientnet_b0(weights=w).features     # 1280 channels
        self.proj = nn.Linear(1280, img_repr)                                       # Eq. (3)
        self.use_env = variant != "M0"
        if self.use_env:                                                            # Eq. (4)
            self.env = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, env_repr), nn.ReLU())
        self.fuse = nn.Linear(img_repr + (env_repr if self.use_env else 0), fusion_w)  # Eq. (5)
        self.cls = nn.Linear(fusion_w, K)                                           # Eq. (6)
        self.sev = nn.Linear(fusion_w, 1)                                           # Eq. (7)

    def encode(self, x):
        return self.features(x)

    def head(self, A, e=None):
        z = A.mean(dim=(2, 3))                                                      # Eq. (2)
        h = F.relu(self.proj(z))
        if self.use_env:
            h = torch.cat([h, self.env(e)], dim=1)
        hF = F.relu(self.fuse(h))
        return self.cls(hF), 1.0 + 4.0 * torch.sigmoid(self.sev(hF)).squeeze(1), hF

    def forward(self, x, e=None):
        return self.head(self.encode(x), e)[:2]

    def set_trainable(self, stage, unfreeze_blocks=3):
        for p in self.features.parameters():
            p.requires_grad = False
        if stage == 2:
            for blk in list(self.features)[-unfreeze_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        if self.frozen_bn:
            self.features.eval()      # batch-norm statistics of the backbone stay at their pretrained values
        return self


# ============================================================================ 5. training
def gpu_augment(x, y, cfg):
    """x float in [0,1], shape B,3,H,W. Rotation, flips and colour jitter on the GPU; stronger for minority classes."""
    B, dev = x.size(0), x.device
    ang = (torch.rand(B, device=dev) * 2 - 1) * cfg["rotation_deg"] * math.pi / 180
    cos, sin = torch.cos(ang), torch.sin(ang)
    sx = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
    sy = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 1] = cos * sx, -sin * sy
    theta[:, 1, 0], theta[:, 1, 1] = sin * sx, cos * sy
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)
    minor = torch.isin(y, torch.tensor(MINORITY, device=dev))
    s = torch.where(minor, cfg["jitter_strong"], cfg["jitter_standard"]).view(B, 1, 1, 1)
    r = lambda: (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * s
    x = x * (1 + r())
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - m) * (1 + r()) + m
    g = x.mean(dim=1, keepdim=True)
    x = (x - g) * (1 + r()) + g
    return x.clamp(0, 1)


def normalise(x):
    m = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


def getx(D, b):
    return D["X"][b].to(D["dev"])


def to_float(xb):
    return xb.permute(0, 3, 1, 2).float() / 255.0


def loss_fn(logits, s_hat, y, grade, mask, w, lam):
    cls = F.cross_entropy(logits, y, weight=w, reduction="sum") / y.size(0)       # Eq. (9), divided by B
    sev = (mask * (s_hat - grade) ** 2).sum() / torch.clamp(mask.sum(), min=1.0)    # Eq. (10)
    return cls + lam * sev, cls, sev                                              # Eq. (11)


@torch.no_grad()
def predict(model, D, idx, E, bs=128):
    model.eval()
    dev = D["dev"]
    L, S = [], []
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        x = normalise(to_float(getx(D, b)))
        e = E[b] if E is not None else None
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            lg, s = model(x, e)
        L.append(lg.float())
        S.append(s.float())
    return torch.cat(L), torch.cat(S)


def macro_f1(y, p):
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, labels=list(range(K)), average="macro", zero_division=0))


def train_variant(variant, seed, D, E, cfg, fusion_w, lr1, ep1, ep2, log, tag):
    """Two-stage training. Returns best state dict (by validation macro-F1) and the history."""
    dev = D["dev"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = Net(variant, fusion_w, cfg["img_repr"], cfg["env_repr"], pretrained=not cfg["smoke"]).to(dev)
    tr, va = D["idx"]["train"], D["idx"]["val"]
    w = D["class_weights"]
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    g = torch.Generator().manual_seed(seed)
    hist, best, best_f1, best_ep, bad, t0 = [], None, -1.0, 0, 0, time.time()
    epoch = 0
    for stage, n_ep, lr in [(1, ep1, lr1), (2, ep2, cfg["lr_stage2"])]:
        model.set_trainable(stage, cfg["unfreeze_blocks"])
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
        for _ in range(n_ep):
            epoch += 1
            model.train()
            perm = torch.randperm(len(tr), generator=g).numpy()
            tot = tc = ts = 0.0
            nb = 0
            for i in range(0, len(perm), cfg["batch_size"]):
                b = tr[perm[i:i + cfg["batch_size"]]]
                if len(b) < 2:
                    continue
                y = D["y"][b]
                x = normalise(gpu_augment(to_float(getx(D, b)), y, cfg))
                e = E[b] if E is not None else None
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    lg, s = model(x, e)
                loss, lc, ls = loss_fn(lg.float(), s.float(), y, D["grade"][b], D["mask"][b], w, cfg["lam"])
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot, tc, ts, nb = tot + loss.item(), tc + lc.item(), ts + ls.item(), nb + 1
            lg, s = predict(model, D, va, E)
            yv = D["y"][va]
            vloss, _, _ = loss_fn(lg, s, yv, D["grade"][va], D["mask"][va], w, cfg["lam"])
            pv = lg.argmax(1)
            f1 = macro_f1(yv.cpu().numpy(), pv.cpu().numpy())
            mv = D["mask"][va] > 0
            vmae = float((s[mv] - D["grade"][va][mv]).abs().mean()) if mv.any() else float("nan")
            hist.append({"epoch": epoch, "stage": stage, "train_loss": tot / nb, "train_cls_loss": tc / nb,
                         "train_sev_loss": ts / nb, "val_loss": float(vloss), "val_acc": float((pv == yv).float().mean()),
                         "val_macro_f1": f1, "val_mae": vmae})
            log(f"{tag} ep{epoch} st{stage} loss={tot / nb:.3f} val_acc={hist[-1]['val_acc']:.3f} val_F1={f1:.3f}")
            if f1 > best_f1:
                best_f1, best_ep, bad = f1, epoch, 0
                best = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            elif stage == 2:
                bad += 1
                if bad >= cfg["patience"]:
                    log(f"{tag} early stop (patience {cfg['patience']})")
                    break
    model.load_state_dict(best)
    return model, best, pd.DataFrame(hist), {"best_epoch": best_ep, "epochs_run": epoch,
                                             "train_minutes": (time.time() - t0) / 60, "best_val_f1": best_f1}


def evaluate(model, D, E, part_name):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    idx = D["idx"][part_name]
    lg, s = predict(model, D, idx, E)
    prob = F.softmax(lg, 1).cpu().numpy()
    y = D["y"][idx].cpu().numpy()
    p = prob.argmax(1)
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=list(range(K)), zero_division=0)
    m = (D["mask"][idx] > 0).cpu().numpy()
    sv, gr = s.cpu().numpy(), D["grade"][idx].cpu().numpy()
    res = {"accuracy": float((p == y).mean()), "macro_f1": float(np.mean(f1)),
           "per_class": {CLASS_NAMES[k]: {"precision": float(pr[k]), "recall": float(rc[k]), "f1": float(f1[k]),
                                          "support": int(sup[k])} for k in range(K)},
           "confusion_matrix": confusion_matrix(y, p, labels=list(range(K))).tolist(),
           "mae": float(np.abs(sv[m] - gr[m]).mean()) if m.any() else None,
           "rmse": float(np.sqrt(((sv[m] - gr[m]) ** 2).mean())) if m.any() else None,
           "n_graded_test": int(m.sum())}
    pred = pd.DataFrame({"image_id": D["ids"][idx.cpu().numpy() if torch.is_tensor(idx) else idx], "true": y, "pred": p,
                         **{f"p_{CLASS_NAMES[k]}": prob[:, k] for k in range(K)}, "s_hat": sv,
                         "grade": np.where(m, gr, np.nan)})
    return res, pred


# ============================================================================ 6. statistics
def conf_macro_f1(conf):
    tp = np.diag(conf).astype(float)
    pr = np.divide(tp, conf.sum(0), out=np.zeros(K), where=conf.sum(0) > 0)
    rc = np.divide(tp, conf.sum(1), out=np.zeros(K), where=conf.sum(1) > 0)
    f1 = np.divide(2 * pr * rc, pr + rc, out=np.zeros(K), where=(pr + rc) > 0)
    return f1.mean()


def bootstrap_stats(preds, B, seed=0):
    """preds: {variant: DataFrame}. Returns percentile CIs of macro-F1 and of M1-M0, M1-M2 differences."""
    rng = np.random.default_rng(seed)
    n = len(preds["M1"])
    y = {v: preds[v]["true"].values for v in preds}
    p = {v: preds[v]["pred"].values for v in preds}
    f = {v: np.zeros(B) for v in preds}
    for b in range(B):
        ii = rng.integers(0, n, n)
        for v in preds:
            conf = np.bincount(y[v][ii] * K + p[v][ii], minlength=K * K).reshape(K, K)
            f[v][b] = conf_macro_f1(conf)
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return ({v: ci(f[v]) for v in f},
            {"M1_minus_M0": ci(f["M1"] - f["M0"]), "M1_minus_M2": ci(f["M1"] - f["M2"])})


def mcnemar(a, b):
    from scipy.stats import binomtest
    ca = (a["pred"].values == a["true"].values)
    cb = (b["pred"].values == b["true"].values)
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())   # a right & b wrong, a wrong & b right
    p = 1.0 if n01 + n10 == 0 else float(binomtest(n01, n01 + n10, 0.5).pvalue)
    return {"b": n01, "c": n10, "p_value": p}


# ============================================================================ 7. explainability
def gradcam(model, x, e, cls=None):
    """Grad-CAM (Selvaraju et al., 2017) on the last convolutional feature map of the image branch."""
    model.eval()
    with torch.enable_grad():
        A = model.encode(x)
        A.retain_grad()
        logits, _, _ = model.head(A, e)
        c = int(logits.argmax(1)) if cls is None else cls
        model.zero_grad()
        logits[0, c].backward()
    alpha = A.grad.mean(dim=(2, 3), keepdim=True)                                   # Eq. (12)
    cam = F.relu((alpha * A).sum(1, keepdim=True))                                  # Eq. (13)
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / cam.max() if cam.max() > 0 else cam
    return cam.detach().cpu().numpy(), c, F.softmax(logits, 1)[0].detach().cpu().numpy()


def cam_region(cam):
    """Describe where the Grad-CAM mass lies, from fixed rules on the heatmap."""
    H, W = cam.shape
    tot = cam.sum() + 1e-9
    centre = cam[H // 4: 3 * H // 4, W // 4: 3 * W // 4].sum() / tot
    top = cam[: H // 2].sum() / tot
    if centre >= 0.5:
        return "the centre of the leaf"
    if top >= 0.6:
        return "the upper part of the leaf"
    if top <= 0.4:
        return "the lower part of the leaf"
    return "the outer parts of the leaf on both sides"


def exact_shap(model, A_or_x, e_row, background, target, device):
    """Exact Shapley values of the four environmental features for the probability of class `target`.
    Image representation is held fixed; absent features are drawn from the background set (interventional)."""
    import itertools
    model.eval()
    nF = e_row.shape[0]
    with torch.no_grad():
        z = A_or_x.mean(dim=(2, 3))
        hI = F.relu(model.proj(z))
        nb = background.shape[0]

        def value(S):
            E = background.clone()
            for j in S:
                E[:, j] = e_row[j]
            h = torch.cat([hI.expand(nb, -1), model.env(E)], dim=1)
            hF = F.relu(model.fuse(h))
            return F.softmax(model.cls(hF), 1)[:, target].mean().item()

        cache = {}
        for r in range(nF + 1):
            for S in itertools.combinations(range(nF), r):
                cache[S] = value(S)
        phi = np.zeros(nF)
        for j in range(nF):
            others = [k for k in range(nF) if k != j]
            for r in range(nF):
                for S in itertools.combinations(others, r):
                    wgt = math.factorial(r) * math.factorial(nF - r - 1) / math.factorial(nF)
                    phi[j] += wgt * (cache[tuple(sorted(S + (j,)))] - cache[S])
    return phi, cache[()], cache[tuple(range(nF))]


def explanation_text(cls_name, conf, grade, region, phi, base, sev_available):
    pos, neg = int(np.argmax(phi)), int(np.argmin(phi))
    parts = [f"The model predicts {CLASS_FULL[cls_name]} ({cls_name}) with a probability of {conf * 100:.0f}%."]
    if sev_available:
        parts.append(f"The estimated severity grade is {grade:.1f} on a scale from 1 to 5.")
    parts.append(f"In the image, the model paid most attention to {region}.")
    if phi[pos] > 0:
        parts.append(f"Among the weather variables, {ENV_LABELS[ENV_NAMES[pos]].lower()} raised the predicted "
                     f"probability most (by {phi[pos] * 100:.1f} percentage points)")
    else:
        parts.append(f"No weather variable raised the predicted probability ({ENV_LABELS[ENV_NAMES[pos]].lower()} "
                     f"changed it by {phi[pos] * 100:.1f} percentage points)")
    if phi[neg] < 0:
        parts[-1] += (f", and {ENV_LABELS[ENV_NAMES[neg]].lower()} lowered it most "
                      f"(by {abs(phi[neg]) * 100:.1f} percentage points).")
    else:
        parts[-1] += "."
    parts.append("The weather values are a regional and seasonal approximation, so this describes how the model "
                 "used its inputs and does not show that the weather caused the disease.")
    return " ".join(parts)


def overlay(img, cam, alpha=0.45):
    cm = plt.get_cmap("jet")(cam)[..., :3]
    return np.clip((1 - alpha) * img + alpha * cm, 0, 1)


def run_explainability(model, D, E_m1, prof_scaled, cfg, out, severity_ok, log, smoke):
    ex_dir = os.path.join(out, "explainability")
    os.makedirs(ex_dir, exist_ok=True)
    dev = D["dev"]
    model.eval()
    test_idx = D["idx"]["test"].cpu().numpy()
    tidx = torch.as_tensor(test_idx, device=dev)
    lg, sv = predict(model, D, tidx, E_m1)
    pred = lg.argmax(1).cpu().numpy()
    true = D["y"][tidx].cpu().numpy()
    rng = np.random.default_rng(11)
    # background: training profiles (scaled)
    tr = D["idx"]["train"]
    bg_pool = E_m1[tr]
    bg = bg_pool[torch.as_tensor(rng.choice(len(bg_pool), size=min(cfg["shap_background"], len(bg_pool)),
                                           replace=False), device=dev)]
    # ---- SHAP on a random sample of test images
    ns = min(cfg["shap_samples"], len(test_idx))
    pick = rng.choice(len(test_idx), size=ns, replace=False)
    rows, mean_abs_acc = [], np.zeros(4)
    for r in pick:
        gi = test_idx[r]
        x = normalise(to_float(getx(D, slice(gi, gi + 1))))
        with torch.no_grad():
            A = model.encode(x)
        e = E_m1[gi]
        phi, base, fx = exact_shap(model, A, e, bg, int(pred[r]), dev)
        rows.append({"image_id": D["ids"][gi], "pred": CLASS_NAMES[pred[r]], "base": base, "fx": fx,
                     **{f"phi_{n}": phi[j] for j, n in enumerate(ENV_NAMES)},
                     **{f"val_{n}": float(e[j]) for j, n in enumerate(ENV_NAMES)}})
    shap_df = pd.DataFrame(rows)
    shap_df.to_csv(os.path.join(ex_dir, "shap_values.csv"), index=False)
    mean_abs = {n: float(shap_df[f"phi_{n}"].abs().mean()) for n in ENV_NAMES}
    json.dump(mean_abs, open(os.path.join(ex_dir, "shap_mean_abs.json"), "w"), indent=2)
    additivity = float((shap_df[[f"phi_{n}" for n in ENV_NAMES]].sum(1) - (shap_df["fx"] - shap_df["base"])).abs().max())
    # ---- cross-check against the shap package (KernelExplainer) on a few images
    check = {"additivity_max_abs_error": additivity, "kernel_explainer_checked": 0, "max_abs_diff": None}
    try:
        import shap
        diffs = []
        for r in pick[:3]:
            gi = test_idx[r]
            x = normalise(to_float(getx(D, slice(gi, gi + 1))))
            with torch.no_grad():
                A = model.encode(x)
                hI = F.relu(model.proj(A.mean(dim=(2, 3))))

            def f(Z, hI=hI, c=int(pred[r])):
                Z = torch.as_tensor(Z, dtype=torch.float32, device=dev)
                with torch.no_grad():
                    h = torch.cat([hI.expand(Z.shape[0], -1), model.env(Z)], dim=1)
                    return F.softmax(model.cls(F.relu(model.fuse(h))), 1)[:, c].cpu().numpy()

            ke = shap.KernelExplainer(f, bg.cpu().numpy())
            v = np.asarray(ke.shap_values(E_m1[gi].cpu().numpy()[None], nsamples=200, silent=True)).reshape(-1)[:4]
            mine = shap_df[shap_df["image_id"] == D["ids"][gi]][[f"phi_{n}" for n in ENV_NAMES]].values[0]
            diffs.append(float(np.abs(v - mine).max()))
        check.update(kernel_explainer_checked=len(diffs), max_abs_diff=max(diffs))
    except Exception as ex:  # shap version problems must not stop the run
        check["note"] = f"KernelExplainer check skipped: {type(ex).__name__}"
    json.dump(check, open(os.path.join(ex_dir, "shap_check.json"), "w"), indent=2)
    log("SHAP exact values done; check:", check)
    # ---- figures: bar and strip plots
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.barh([ENV_LABELS[n] for n in ENV_NAMES][::-1], [mean_abs[n] for n in ENV_NAMES][::-1], color=PAL["M1"])
    ax.set_xlabel("Mean absolute SHAP value (probability of predicted class)")
    save_fig(fig, os.path.join(out, "figures", "shap_summary.png"), smoke)
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    for j, n in enumerate(ENV_NAMES):
        v = shap_df[f"val_{n}"].values
        ax.scatter(shap_df[f"phi_{n}"], j + (np.random.default_rng(j).random(len(v)) - 0.5) * 0.5, c=v, cmap="coolwarm",
                   s=9, vmin=0, vmax=1)
    ax.set_yticks(range(4))
    ax.set_yticklabels([ENV_LABELS[n] for n in ENV_NAMES])
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel("SHAP value (colour: scaled feature value, low to high)")
    save_fig(fig, os.path.join(out, "figures", "shap_beeswarm.png"), smoke)
    # ---- Grad-CAM samples: up to 2 correct and 2 wrong per class
    chosen = []
    for c in range(K):
        ok = [i for i in range(len(test_idx)) if true[i] == c and pred[i] == c]
        bad = [i for i in range(len(test_idx)) if true[i] == c and pred[i] != c]
        chosen += [(i, "correct") for i in ok[:2]] + [(i, "wrong") for i in bad[:2]]
    expl, shown = [], []
    for i, kind in chosen:
        gi = test_idx[i]
        x = normalise(to_float(getx(D, slice(gi, gi + 1))))
        e = E_m1[gi:gi + 1]
        cam, c, prob = gradcam(model, x, e)
        phi, base, fx = exact_shap(model, model.encode(x).detach(), E_m1[gi], bg, c, dev)
        g_est = float(sv[i])
        txt = explanation_text(CLASS_NAMES[c], float(prob[c]), g_est, cam_region(cam), phi, base, severity_ok)
        img = D["X"][gi].cpu().numpy() / 255.0
        shown.append((img, cam, f"True: {CLASS_NAMES[true[i]]}  Pred: {CLASS_NAMES[c]} ({kind})"))
        expl.append({"image_id": D["ids"][gi], "true": CLASS_NAMES[true[i]], "pred": CLASS_NAMES[c], "kind": kind,
                     "confidence": float(prob[c]), "severity_estimate": g_est if severity_ok else None,
                     "region": cam_region(cam), "phi": {n: float(phi[j]) for j, n in enumerate(ENV_NAMES)},
                     "text": txt})
    json.dump(expl, open(os.path.join(ex_dir, "explanations.json"), "w"), indent=2)
    if shown:
        cols = 4
        rows = math.ceil(len(shown) / cols)
        fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 2.4, rows * 4.9))
        axes = np.atleast_2d(axes)
        for ax in axes.ravel():
            ax.axis("off")
        for k, (img, cam, cap) in enumerate(shown):
            r, c = divmod(k, cols)
            axes[2 * r, c].imshow(img)
            axes[2 * r, c].set_title(cap, fontsize=6.5)
            axes[2 * r + 1, c].imshow(overlay(img, cam))
        save_fig(fig, os.path.join(out, "figures", "gradcam_samples.png"), smoke)
    # ---- panel: original | Grad-CAM | text for the first three samples
    if expl:
        fig, axes = plt.subplots(3, 3, figsize=(10.5, 7.2), gridspec_kw={"width_ratios": [1, 1, 2.3]})
        for r in range(3):
            for ax in axes[r]:
                ax.axis("off")
            if r < len(shown):
                img, cam, cap = shown[r]
                axes[r, 0].imshow(img)
                axes[r, 1].imshow(overlay(img, cam))
                import textwrap
                axes[r, 2].text(0, 0.95, cap + "\n\n" + "\n".join(textwrap.wrap(expl[r]["text"], 62)),
                                va="top", fontsize=8)
        save_fig(fig, os.path.join(out, "figures", "explanation_panel.png"), smoke)
    return {"mean_abs": mean_abs, "check": check}


# ============================================================================ 8. figures for results
def make_basic_figures(df, part, prof, out, smoke, summ):
    fd = os.path.join(out, "figures")
    os.makedirs(fd, exist_ok=True)
    counts = [int((df["label"] == k).sum()) for k in range(K)]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    bars = ax.bar(CLASS_NAMES, counts, color="#0072B2")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c:,}\n({100 * c / sum(counts):.1f}%)", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Number of images")
    ax.set_ylim(0, max(counts) * 1.2)
    save_fig(fig, os.path.join(fd, "eda_class_distribution.png"), smoke)
    fig, axes = plt.subplots(2, K, figsize=(K * 2.0, 4.3))
    rng = np.random.default_rng(3)
    X = np.load(os.path.join(summ["cache_dir"], f"images_{summ['img_size']}.npy"), mmap_mode="r")
    for c in range(K):
        ids = np.where(df["label"].values == c)[0]
        for r, i in enumerate(rng.choice(ids, size=2, replace=False)):
            axes[r, c].imshow(X[i])
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(CLASS_NAMES[c], fontsize=9)
    save_fig(fig, os.path.join(fd, "sample_images.png"), smoke)
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    M = prof[ENV_NAMES].values
    Mn = (M - M.min(0)) / (M.max(0) - M.min(0) + 1e-9)
    im = ax.imshow(Mn, cmap="viridis", aspect="auto")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["T2M", "RH2M", "Rain", "Solar"])
    ax.set_yticks(range(len(prof)))
    ax.set_yticklabels([f"{z} {s}" for z, s in zip(prof["zone"], prof["season"])], fontsize=7)
    for i in range(len(prof)):
        for j in range(4):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=6.5, color="white" if Mn[i, j] < 0.6 else "black")
    fig.colorbar(im, ax=ax, label="Min-Max scaled value", fraction=0.04)
    save_fig(fig, os.path.join(fd, "weather_profiles.png"), smoke)


def make_result_figures(runs, hist, summary, out, smoke):
    fd = os.path.join(out, "figures")
    for (v, s), h in hist.items():
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
        axes[0].plot(h["epoch"], h["train_loss"], label="Training loss", color=PAL[v])
        axes[0].plot(h["epoch"], h["val_loss"], "--", label="Validation loss", color="black")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend(frameon=False, fontsize=8)
        axes[1].plot(h["epoch"], h["val_macro_f1"], color=PAL[v])
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Validation macro F1-score")
        st = h[h["stage"] == 2]["epoch"].min() if (h["stage"] == 2).any() else None
        for ax in axes:
            if st:
                ax.axvline(st - 0.5, color="grey", lw=0.6, ls=":")
        save_fig(fig, os.path.join(fd, f"curves_{v}_seed{s}.png"), smoke)
    first = sorted({s for (_, s) in runs})[0]
    for v in ["M0", "M1", "M2"]:
        conf = np.array(runs[(v, first)]["confusion_matrix"], dtype=float)
        cn = conf / conf.sum(1, keepdims=True).clip(1)
        fig, ax = plt.subplots(figsize=(4.4, 3.8))
        im = ax.imshow(cn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(K))
        ax.set_xticklabels(CLASS_NAMES, fontsize=8)
        ax.set_yticks(range(K))
        ax.set_yticklabels(CLASS_NAMES, fontsize=8)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{int(conf[i, j])}", ha="center", va="center", fontsize=8, color="white" if cn[i, j] > 0.5 else "black")
        fig.colorbar(im, ax=ax, label="Row-normalised", fraction=0.046)
        save_fig(fig, os.path.join(fd, f"confusion_{v}.png"), smoke)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for k, v in enumerate(["M0", "M1", "M2"]):
        vals = summary["variants"][v]["macro_f1"]["values"]
        ax.bar(k, summary["variants"][v]["macro_f1"]["mean"], yerr=summary["variants"][v]["macro_f1"]["sd"], color=PAL[v],
               capsize=4, alpha=0.85)
        ax.scatter([k] * len(vals), vals, color="black", s=12, zorder=3)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["M0 image-only", "M1 multimodal", "M2 shuffled"], fontsize=8)
    ax.set_ylabel("Test macro F1-score")
    lo = min(min(summary["variants"][v]["macro_f1"]["values"]) for v in summary["variants"])
    ax.set_ylim(max(0, lo - 0.08), 1.0)
    save_fig(fig, os.path.join(fd, "comparison_macroF1.png"), smoke)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    wbar = 0.26
    for k, v in enumerate(["M0", "M1", "M2"]):
        m = [summary["variants"][v]["per_class_f1"][c]["mean"] for c in CLASS_NAMES]
        sd = [summary["variants"][v]["per_class_f1"][c]["sd"] for c in CLASS_NAMES]
        ax.bar(np.arange(K) + (k - 1) * wbar, m, wbar, yerr=sd, color=PAL[v], label=v, capsize=2)
    ax.set_xticks(range(K))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("Test F1-score")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    save_fig(fig, os.path.join(fd, "per_class_f1.png"), smoke)


def severity_scatter(pred, out, smoke):
    g = pred.dropna(subset=["grade"])
    if len(g) < 3:
        return
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    jit = (np.random.default_rng(0).random(len(g)) - 0.5) * 0.15
    ax.scatter(g["grade"] + jit, g["s_hat"], s=14, color=PAL["M1"], alpha=0.8)
    ax.plot([1, 5], [1, 5], color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Visual grade")
    ax.set_ylabel("Predicted severity score")
    save_fig(fig, os.path.join(out, "figures", "severity_scatter.png"), smoke)


# ============================================================================ 9. main
def aggregate(runs, seeds):
    summary = {"variants": {}}
    for v in ["M0", "M1", "M2"]:
        r = [runs[(v, s)] for s in seeds]
        agg = lambda key: {"mean": float(np.mean([x[key] for x in r])) if r[0][key] is not None else None,
                           "sd": float(np.std([x[key] for x in r], ddof=1)) if len(r) > 1 and r[0][key] is not None else 0.0,
                           "values": [x[key] for x in r]}
        d = {"accuracy": agg("accuracy"), "macro_f1": agg("macro_f1"), "mae": agg("mae"), "rmse": agg("rmse")}
        for met in ["precision", "recall", "f1"]:
            d[f"per_class_{met}"] = {c: {"mean": float(np.mean([x["per_class"][c][met] for x in r])),
                                         "sd": float(np.std([x["per_class"][c][met] for x in r], ddof=1)) if len(r) > 1 else 0.0}
                                     for c in CLASS_NAMES}
        summary["variants"][v] = d
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="folder with train.csv and train_images/ (auto-detected under /kaggle/input)")
    ap.add_argument("--out", default="/kaggle/working/results")
    ap.add_argument("--weather-dir", default=None, help="folder with weather_profiles.csv and weather_meta.json")
    ap.add_argument("--quick", action="store_true", help="one seed, no tuning, fewer epochs")
    ap.add_argument("--make-grading-sheet", action="store_true", help="only prepare the severity grading task, then stop")
    ap.add_argument("--grades-text", default="", help="pasted grades, one 'code grade' pair per line")
    ap.add_argument("--grades-file", default=None, help="text file with one 'code grade' pair per line")
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic fixture; NOT real results")
    ap.add_argument("--mock-weather", action="store_true", help="fake weather (only allowed with --smoke)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cfg = copy.deepcopy(DEFAULTS)
    cfg["smoke"] = a.smoke
    if a.grades_file:
        a.grades_text = open(a.grades_file, encoding="utf-8").read()
    if a.mock_weather and not a.smoke:
        sys.exit("--mock-weather is only allowed with --smoke. Real runs never use invented weather.")
    if a.smoke:
        cfg.update(img_size=96, epochs_stage1=1, epochs_stage2=2, patience=2, seeds=[1, 2], n_bootstrap=200,
                   shap_samples=12, shap_background=20, tune_epochs=(1, 1), tune_grid={"lr_stage1": [1e-3], "fusion_w": [128, 64]},
                   grade_per_class={"train": 6, "val": 3, "test": 4})
    if a.quick:
        cfg.update(seeds=[42], epochs_stage2=6)
    if a.seeds:
        cfg["seeds"] = a.seeds
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    log = Logger(os.path.join(a.out, "run_log.txt"))
    t_start = dt.datetime.now()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "", "| smoke" if a.smoke else "")

    # ---- data
    a.data = find_data_dir(a.data, log)
    df, X_np, load_summ = prepare_images(a.data, a.out, cfg, log, limit=a.limit)
    labels = df["label"].values.astype(int)
    part = stratified_split(labels)
    part_sizes = {p: int((part == p).sum()) for p in ["train", "val", "test"]}
    cc = lambda m: {CLASS_NAMES[k]: int(((labels == k) & m).sum()) for k in range(K)}
    ntr = np.array([(labels[part == "train"] == k).sum() for k in range(K)])
    cw = len(labels[part == "train"]) / (K * ntr)
    ds = {k: v for k, v in load_summ.items() if k != "cache_dir"}
    ds.update({"class_names": CLASS_NAMES, "class_counts_total": cc(np.ones(len(labels), bool)),
               "class_counts_by_partition": {p: cc(part == p) for p in ["train", "val", "test"]},
               "partition_sizes": part_sizes, "class_weights": {CLASS_NAMES[k]: float(cw[k]) for k in range(K)}})
    json.dump(ds, open(os.path.join(a.out, "dataset_summary.json"), "w"), indent=2)

    # ---- severity grading task
    sheet, gdir = make_grading_sheet(df, part, a.out, cfg, log, a.smoke)
    if a.make_grading_sheet:
        draw_montages(sheet, a.data, gdir, a.smoke)
        log(f"grading sheet ({len(sheet)} images) and montages written to {gdir}")
        return
    grade, mask, sev_summ = load_grades(a.out, df, part, cfg, a.grades_text, log)
    json.dump(sev_summ, open(os.path.join(a.out, "severity_summary.json"), "w"), indent=2)

    # ---- environmental data
    prof, wmeta = get_profiles(a.out, a.weather_dir, a.mock_weather, log)
    rng = np.random.default_rng(SPLIT_SEED + 3)
    pid = rng.integers(0, len(prof), size=len(df))
    raw = prof[ENV_NAMES].values[pid].astype(np.float32)
    tr_mask = part == "train"
    mn, mx = raw[tr_mask].min(0), raw[tr_mask].max(0)                                  # Eq. (1): training partition only
    scaled = (raw - mn) / np.where(mx - mn == 0, 1, mx - mn)
    json.dump({"min": mn.tolist(), "max": mx.tolist(), "features": ENV_NAMES}, open(os.path.join(a.out, "minmax.json"), "w"))
    scaled_shuf = scaled.copy()
    rs = np.random.default_rng(SPLIT_SEED + 4)
    for p in ["train", "val", "test"]:
        ii = np.where(part == p)[0]
        scaled_shuf[ii] = scaled[rs.permutation(ii)]
    pd.DataFrame({"image_id": df["image_id"], "label": labels, "partition": part, "profile_id": pid}).to_csv(
        os.path.join(a.out, "splits.csv"), index=False)
    make_basic_figures(df, part, prof, a.out, a.smoke, {"img_size": cfg["img_size"], "cache_dir": load_summ["cache_dir"]})

    # ---- tensors on device
    X = torch.from_numpy(np.ascontiguousarray(X_np))
    if dev.type == "cuda" and X.numel() < 0.5 * torch.cuda.get_device_properties(0).total_memory:
        X = X.to(dev)
    D = {"dev": dev, "X": X, "y": torch.as_tensor(labels, device=dev),
         "grade": torch.as_tensor(grade, device=dev), "mask": torch.as_tensor(mask, device=dev),
         "ids": df["image_id"].values, "class_weights": torch.as_tensor(cw, dtype=torch.float32, device=dev),
         "idx": {p: torch.as_tensor(np.where(part == p)[0], device=X.device if X.is_cuda else "cpu") for p in ["train", "val", "test"]}}
    D["X_dev"] = X.device
    E_real = torch.as_tensor(scaled, device=dev)
    E_shuf = torch.as_tensor(scaled_shuf, device=dev)
    if X.device.type == "cpu" and dev.type == "cuda":
        pass
    envs = {"M0": None, "M1": E_real, "M2": E_shuf}

    # ---- tuning (M1, first seed, validation macro-F1)
    fusion_w, lr1, tuning = cfg["fusion_w"], cfg["lr_stage1"], {"ran": False, "table": [], "chosen": {}}
    if not a.quick:
        tuning["ran"] = True
        best = -1
        for lr in cfg["tune_grid"]["lr_stage1"]:
            for fw in cfg["tune_grid"]["fusion_w"]:
                _, _, h, info = train_variant("M1", cfg["seeds"][0], D, envs["M1"], cfg, fw, lr, cfg["tune_epochs"][0],
                                              cfg["tune_epochs"][1], log, f"tune lr={lr} fw={fw}")
                tuning["table"].append({"lr_stage1": lr, "fusion_w": fw, "val_macro_f1": info["best_val_f1"]})
                if info["best_val_f1"] > best:
                    best, fusion_w, lr1 = info["best_val_f1"], fw, lr
        tuning["chosen"] = {"lr_stage1": lr1, "fusion_w": fusion_w, "batch_size": cfg["batch_size"]}
        pd.DataFrame(tuning["table"]).to_csv(os.path.join(a.out, "tuning_results.csv"), index=False)
    else:
        tuning["chosen"] = {"lr_stage1": lr1, "fusion_w": fusion_w, "batch_size": cfg["batch_size"]}
    log("chosen settings:", tuning["chosen"])

    # ---- main runs
    runs, hist, preds, val_f1 = {}, {}, {}, {}
    m1_models = {}
    for seed in cfg["seeds"]:
        for v in ["M0", "M1", "M2"]:
            tag = f"{v}_seed{seed}"
            rd = os.path.join(a.out, "runs", tag)
            os.makedirs(rd, exist_ok=True)
            model, state, h, info = train_variant(v, seed, D, envs[v], cfg, fusion_w, lr1, cfg["epochs_stage1"],
                                                  cfg["epochs_stage2"], log, tag)
            res, pr = evaluate(model, D, envs[v], "test")
            res.update({k: info[k] for k in ["best_epoch", "epochs_run", "train_minutes"]})
            h.to_csv(os.path.join(rd, "history.csv"), index=False)
            pr.to_csv(os.path.join(rd, "predictions.csv"), index=False)
            json.dump(res, open(os.path.join(rd, "test_metrics.json"), "w"), indent=2)
            if v == "M1":
                torch.save({"state_dict": state, "fusion_w": fusion_w, "img_repr": cfg["img_repr"],
                            "env_repr": cfg["env_repr"], "img_size": cfg["img_size"], "smoke": a.smoke,
                            "seed": seed, "best_val_f1": info["best_val_f1"]}, os.path.join(rd, "best.pt"))
                val_f1[seed] = info["best_val_f1"]
            runs[(v, seed)], hist[(v, seed)], preds[(v, seed)] = res, h, pr
            log(f"{tag}: test acc={res['accuracy']:.4f} macro-F1={res['macro_f1']:.4f}")
            del model
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    # ---- summary and statistics
    summary = aggregate(runs, cfg["seeds"])
    first = cfg["seeds"][0]
    p1 = {v: preds[(v, first)] for v in ["M0", "M1", "M2"]}
    ci, dci = bootstrap_stats(p1, cfg["n_bootstrap"])
    summary["tests"] = {"mcnemar_M1_vs_M0": mcnemar(p1["M1"], p1["M0"]), "mcnemar_M1_vs_M2": mcnemar(p1["M1"], p1["M2"]),
                        "bootstrap_macro_f1_ci": ci, "bootstrap_diff_ci": dci, "first_seed": first}
    best_seed = max(val_f1, key=val_f1.get)
    summary.update({"tuning": tuning, "seeds_used": cfg["seeds"], "best_m1_seed": int(best_seed)})
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    make_result_figures(runs, hist, summary, a.out, a.smoke)
    severity_scatter(preds[("M1", best_seed)], a.out, a.smoke)

    # ---- explainability for the best M1 model
    ck = torch.load(os.path.join(a.out, "runs", f"M1_seed{best_seed}", "best.pt"), map_location=dev)
    m1 = Net("M1", ck["fusion_w"], ck["img_repr"], ck["env_repr"], pretrained=False).to(dev)
    m1.load_state_dict(ck["state_dict"])
    if X.device.type == "cpu":
        pass
    ex = run_explainability(m1, D, E_real, None, cfg, a.out, sev_summ["available"], log, a.smoke)

    t_end = dt.datetime.now()
    versions = {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
                "numpy": np.__version__, "pandas": pd.__version__}
    try:
        import shap
        versions["shap"] = shap.__version__
    except Exception:
        pass
    json.dump({"smoke": a.smoke, "versions": versions, "device": dev.type,
               "gpu_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else None, "seeds": cfg["seeds"],
               "hyperparameters": {k: v for k, v in cfg.items() if k != "smoke"} | {"final": tuning["chosen"]},
               "start_time": t_start.isoformat(), "end_time": t_end.isoformat(),
               "total_minutes": (t_end - t_start).total_seconds() / 60}, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    if a.smoke:
        open(os.path.join(a.out, "SMOKE_TEST_NOT_REAL_RESULTS.txt"), "w").write(SMOKE_BANNER + "\n")
    shutil.rmtree(os.path.join(a.out, "cache"), ignore_errors=True) if False else None
    zpath = os.path.join(os.path.dirname(a.out.rstrip("/")) or ".", "results.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(a.out):
            if os.sep + "cache" in root:
                continue
            for f in files:
                z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), os.path.dirname(a.out.rstrip("/"))))
    log("finished; results zipped to", zpath, f"({(t_end - t_start).total_seconds() / 60:.1f} min)")


if __name__ == "__main__":
    main()
