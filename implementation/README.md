# Cassava experiment: how to run it

Files
- `cassava_experiment.ipynb`  Kaggle notebook (contains the script; upload this one file).
- `cassava_experiment.py`     the same script for running from a terminal.
- `fetch_weather.py`          downloads NASA POWER data on your own computer (use if Kaggle Internet is unavailable).
- `app.py`                    Streamlit prototype (upload a leaf image, get class, severity, Grad-CAM, SHAP, explanation).
- `tests_smoke.py`, `make_smoke_fixture.py`  code tests on a tiny SYNTHETIC fixture (not cassava data).
- `RESULTS_SCHEMA.md`         list of every output file.

## On Kaggle (recommended)
1. Join the competition "Cassava Leaf Disease Classification" and accept the rules.
2. New Notebook -> File -> Import Notebook -> upload `cassava_experiment.ipynb`.
3. Right-hand panel: Add Input -> the competition; Accelerator GPU T4 x2; Internet On.
4. Run cells A to B. Cell C (optional) makes the severity grading sheet; grade the numbered leaves; paste grades in cell D.
5. Cell E: `QUICK = False` for the full study (3 seeds + small tuning grid), `QUICK = True` for a fast single-seed run.
6. Download `/kaggle/working/results.zip` from the Output tab and send it back for Chapters 4 and 5.

No Internet on Kaggle? On your own computer run `python fetch_weather.py --out weather_dir`, upload the folder as a Kaggle dataset,
add it as an input, and set `WEATHER_DIR` in cell E. Pretrained weights also need Internet: add the torchvision
EfficientNet-B0 weights file as an input, or turn Internet on.

## Streamlit prototype (on your computer, CPU is enough)
    pip install -r requirements.txt
    unzip results.zip            # gives a folder `results`
    streamlit run app.py -- --results results
Use `results/runs/M1_seed*/best.pt` (kept in results.zip). The app picks the M1 seed with the best validation macro F1-score.

## What the run does (Chapter 3)
Cleans and caches the images (224x224), splits 80/10/10 stratified, builds 16 weather profiles (4 Uganda zones x 4 seasons) from
NASA POWER, assigns profiles to images at random (the data have no location or date), trains M0, M1 and M2 (EfficientNet-B0,
two stages, class-weighted cross-entropy plus masked severity MSE, Adam), evaluates on the held-out test set, runs McNemar tests and
bootstrap intervals, and produces Grad-CAM, exact SHAP values for the four weather features and a plain-language explanation.

## Honest notes
- Runtimes depend on the GPU; the code could not be timed on a GPU in the sandbox where it was written. Expect tens of minutes
  per variant-seed on a T4 for the full setting; use `QUICK = True` if time is short.
- The weather profiles are regional and seasonal approximations; M1 may not beat M0. That is a valid result and is reported as is.
- Severity results exist only if you grade leaves. Without grades the severity head is not trained.
- The SHAP values are computed exactly (all 16 feature subsets) and are cross-checked against `shap.KernelExplainer` on three images
  (see `explainability/shap_check.json`).
