"""Streamlit prototype: upload a cassava leaf photo, choose a region and season, get the predicted disease,
severity grade, Grad-CAM heatmap, SHAP values for the weather features and a plain-language explanation.

Run locally:  streamlit run app.py -- --results path/to/results
The results folder is the one produced by cassava_experiment.py (needs runs/M1_seed*/best.pt, minmax.json,
weather_profiles.csv, severity_summary.json, splits.csv).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image

import cassava_experiment as ce


def load_bundle(results):
    ck_path = sorted(glob.glob(os.path.join(results, "runs", "M1_seed*", "best.pt")),
                     key=lambda p: torch.load(p, map_location="cpu")["best_val_f1"])[-1]
    ck = torch.load(ck_path, map_location="cpu")
    model = ce.Net("M1", ck["fusion_w"], ck["img_repr"], ck["env_repr"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    mm = json.load(open(os.path.join(results, "minmax.json")))
    prof = pd.read_csv(os.path.join(results, "weather_profiles.csv"))
    sev = json.load(open(os.path.join(results, "severity_summary.json")))["available"]
    splits = pd.read_csv(os.path.join(results, "splits.csv"))
    bg_ids = splits[splits["partition"] == "train"]["profile_id"].values
    raw = prof[ce.ENV_NAMES].values.astype(np.float32)
    mn, mx = np.array(mm["min"], dtype=np.float32), np.array(mm["max"], dtype=np.float32)
    scaled = (raw - mn) / np.where(mx - mn == 0, 1, mx - mn)
    rng = np.random.default_rng(0)
    bg = torch.as_tensor(scaled[rng.choice(bg_ids, size=min(100, len(bg_ids)), replace=False)])
    return {"model": model, "prof": prof, "scaled": scaled, "bg": bg, "sev": sev, "size": ck["img_size"],
            "smoke": bool(ck.get("smoke", False)), "ckpt": ck_path}


def diagnose(bundle, image, zone, season):
    """Run the framework of Algorithm 3 on one PIL image. Returns a dict of outputs."""
    model, size = bundle["model"], bundle["size"]
    img = image.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    x = ce.normalise(torch.from_numpy(arr.copy()).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0)
    row = bundle["prof"][(bundle["prof"]["zone"] == zone) & (bundle["prof"]["season"] == season)].index[0]
    e = torch.as_tensor(bundle["scaled"][row]).unsqueeze(0)
    cam, c, prob = ce.gradcam(model, x, e)
    with torch.no_grad():
        A = model.encode(x)
        _, s, _ = model.head(A, e)
    phi, base, fx = ce.exact_shap(model, A, e[0], bundle["bg"], c, torch.device("cpu"))
    region = ce.cam_region(cam)
    text = ce.explanation_text(ce.CLASS_NAMES[c], float(prob[c]), float(s), region, phi, base, bundle["sev"])
    return {"cls": c, "prob": prob, "severity": float(s), "cam": cam, "img": arr / 255.0, "phi": phi,
            "region": region, "text": text}


def ui():
    import streamlit as st
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.environ.get("RESULTS_DIR", "results"))
    args, _ = ap.parse_known_args(sys.argv[1:])
    st.set_page_config(page_title="Cassava leaf diagnosis", layout="wide")
    st.title("Cassava leaf disease diagnosis (research prototype)")
    bundle = st.cache_resource(load_bundle)(args.results)
    if bundle["smoke"]:
        st.error("SMOKE TEST MODEL - NOT REAL RESULTS")
    st.warning("Research prototype. The weather profile is a regional and seasonal approximation, not a measurement "
               "taken at the field. Do not use this tool as the only basis for treatment decisions.")
    zones = sorted(bundle["prof"]["zone"].unique())
    seasons = ["MAM", "JJA", "SON", "DJF"]
    c1, c2 = st.sidebar.columns(2)
    zone = st.sidebar.selectbox("Region", zones)
    season = st.sidebar.selectbox("Season", seasons)
    up = st.file_uploader("Upload a photo of a cassava leaf", type=["jpg", "jpeg", "png"])
    if up is None:
        st.info("Upload a leaf photo to start.")
        return
    try:
        image = Image.open(up)
        image.load()
    except Exception:
        st.error("The file could not be read as an image. Please upload a JPEG or PNG photo of a cassava leaf.")
        return
    out = diagnose(bundle, image, zone, season)
    col1, col2 = st.columns(2)
    col1.image(out["img"], caption="Uploaded image", use_container_width=True)
    col2.image(ce.overlay(out["img"], out["cam"]), caption="Grad-CAM heatmap for the predicted class",
               use_container_width=True)
    st.subheader(f"Prediction: {ce.CLASS_FULL[ce.CLASS_NAMES[out['cls']]]} ({out['prob'][out['cls']] * 100:.0f}%)")
    if bundle["sev"]:
        st.metric("Estimated severity grade (1 to 5)", f"{out['severity']:.1f}")
    else:
        st.caption("Severity head was not trained (no severity grades were available).")
    st.bar_chart(pd.DataFrame({"probability": out["prob"]}, index=ce.CLASS_NAMES))
    st.caption("Effect of each weather variable on the probability of the predicted class (SHAP value)")
    st.bar_chart(pd.DataFrame({"SHAP value": out["phi"]}, index=[ce.ENV_LABELS[n] for n in ce.ENV_NAMES]))
    st.write(out["text"])


if __name__ == "__main__":
    ui()
