"""Unit, integration and performance tests of the Streamlit prototype (app.py) and of the functions it uses.

    python test_prototype.py --results results --images leaves --out test_report

--results  folder produced by cassava_experiment.py (needs runs/M1_seed*/best.pt, minmax.json, weather_profiles.csv ...)
--images   folder with a few leaf photographs (jpg/png). If omitted, synthetic images are used and the report says so.
--out      folder for test_report.json and the screenshots

Browser tests use Playwright with Chromium (pip install playwright; the browser must be installed).
Nothing in the report is typed by hand: every figure is measured by this script.
"""
import argparse
import glob
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import app as proto                       # noqa: E402
import cassava_experiment as ce           # noqa: E402

CASES = []


def case(cid, kind, desc, expected):
    def deco(fn):
        CASES.append(dict(id=cid, kind=kind, desc=desc, expected=expected, fn=fn))
        return fn
    return deco


def synth(path, size=(640, 480), mode="RGB", seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.integers(40, 200, (size[1], size[0], 3), dtype=np.uint8)
    arr[:, :, 1] = np.clip(arr[:, :, 1] + 40, 0, 255)
    im = Image.fromarray(arr).convert(mode)
    im.save(path)
    return path


# ===================================================================================== unit tests
@case("UT01", "unit", "Normalisation of an image equal to the ImageNet channel means", "Output is zero in every channel (within 1e-5)")
def ut01(c):
    m = torch.tensor(ce.IMAGENET_MEAN).view(1, 3, 1, 1).expand(1, 3, 8, 8)
    out = ce.normalise(m.clone())
    assert out.abs().max() < 1e-5, f"max {out.abs().max():.2e}"


@case("UT02", "unit", "Output shapes of the multimodal network for a batch of 3 images", "Logits 3 x 5 and severity of length 3")
def ut02(c):
    lg, s = c["bundle"]["model"](torch.rand(3, 3, c["bundle"]["size"], c["bundle"]["size"]), torch.rand(3, 4))
    assert tuple(lg.shape) == (3, 5) and tuple(s.shape) == (3,), (lg.shape, s.shape)


@case("UT03", "unit", "Severity score for extreme inputs (very large and very small activations)", "Always between 1 and 5")
def ut03(c):
    m = c["bundle"]["model"]
    with torch.no_grad():
        for scale in (0.0, 1.0, 50.0):
            x = torch.randn(2, 3, c["bundle"]["size"], c["bundle"]["size"]) * scale
            _, s = m(x, torch.rand(2, 4))
            assert float(s.min()) >= 1 and float(s.max()) <= 5, (scale, s)


@case("UT04", "unit", "Class probabilities of one prediction", "Non-negative and sum to 1 (within 1e-5)")
def ut04(c):
    o = proto.diagnose(c["bundle"], c["pil"][0], c["zone"], c["season"])
    assert o["prob"].min() >= 0 and abs(o["prob"].sum() - 1) < 1e-5, o["prob"]


@case("UT05", "unit", "Grad-CAM map for one image", "Same size as the input, values in [0, 1], no NaN")
def ut05(c):
    o = proto.diagnose(c["bundle"], c["pil"][0], c["zone"], c["season"])
    s = c["bundle"]["size"]
    assert o["cam"].shape == (s, s), o["cam"].shape
    assert np.isfinite(o["cam"]).all() and o["cam"].min() >= 0 and o["cam"].max() <= 1 + 1e-6


@case("UT06", "unit", "Additivity of the exact Shapley values", "Base value plus the four values equals the prediction (within 1e-6)")
def ut06(c):
    b = c["bundle"]
    x = ce.normalise(torch.from_numpy(np.asarray(c["pil"][0].convert("RGB").resize((b["size"], b["size"])), dtype=np.uint8).copy())
                     .unsqueeze(0).permute(0, 3, 1, 2).float() / 255)
    e = torch.rand(4)
    with torch.no_grad():
        A = b["model"].encode(x)
    phi, base, fx = ce.exact_shap(b["model"], A, e, b["bg"], 0, torch.device("cpu"))
    assert abs(base + phi.sum() - fx) < 1e-6, abs(base + phi.sum() - fx)


@case("UT07", "unit", "Min-Max scaled weather profiles held by the application", "All 16 profiles lie in [0, 1] for every feature")
def ut07(c):
    s = c["bundle"]["scaled"]
    assert s.shape == (16, 4) and s.min() >= -1e-6 and s.max() <= 1 + 1e-6, (s.shape, s.min(), s.max())


@case("UT08", "unit", "Explanation text for a prediction", "Names the predicted class, gives the probability, and states that the weather is an approximation that does not prove a cause")
def ut08(c):
    t = ce.explanation_text("CMD", 0.91, 3.0, "the centre of the leaf", np.array([0.02, -0.01, 0.0, 0.005]), 0.5, True)
    assert "Cassava Mosaic Disease (CMD)" in t and "91%" in t and "does not show that the weather caused the disease" in t, t


@case("UT09", "unit", "Explanation text when no weather variable raises the probability", "Text states that no variable raised the probability and has no lowering clause error")
def ut09(c):
    t = ce.explanation_text("CBB", 0.5, 2.0, "the lower part of the leaf", np.array([-0.02, -0.01, -0.03, -0.005]), 0.5, False)
    assert "No weather variable raised" in t and "lowered it most" in t and "severity" not in t.lower().split("approximation")[0], t


@case("UT10", "unit", "Handling of images in other colour modes (greyscale, RGBA, palette)", "Each is converted and diagnosed without error")
def ut10(c):
    for mode in ("L", "RGBA", "P"):
        p = synth(os.path.join(c["tmp"], f"mode_{mode}.png"), (200, 150), mode)
        proto.diagnose(c["bundle"], Image.open(p), c["zone"], c["season"])


@case("UT11", "unit", "Handling of very small (16 x 16) and very large (3000 x 2000) images", "Both are resized and diagnosed without error")
def ut11(c):
    for sz, name in (((16, 16), "tiny"), ((3000, 2000), "huge")):
        p = synth(os.path.join(c["tmp"], f"{name}.jpg"), sz)
        proto.diagnose(c["bundle"], Image.open(p), c["zone"], c["season"])


@case("UT12", "unit", "Every region and season combination (4 x 4 = 16) for one image", "All 16 profiles are found and give a valid prediction")
def ut12(c):
    zones = sorted(c["bundle"]["prof"]["zone"].unique())
    n = 0
    for z in zones:
        for s in ("MAM", "JJA", "SON", "DJF"):
            o = proto.diagnose(c["bundle"], c["pil"][0], z, s)
            assert 0 <= o["cls"] < 5
            n += 1
    assert n == 16, n


@case("UT13", "unit", "Repeatability: the same image and profile twice", "Identical class probabilities")
def ut13(c):
    a = proto.diagnose(c["bundle"], c["pil"][0], c["zone"], c["season"])["prob"]
    b = proto.diagnose(c["bundle"], c["pil"][0], c["zone"], c["season"])["prob"]
    assert np.allclose(a, b, atol=1e-6)


@case("UT14", "unit", "Unknown region or season passed to the diagnosis function", "Rejected with an error and not answered silently")
def ut14(c):
    try:
        proto.diagnose(c["bundle"], c["pil"][0], "Nowhere", "MAM")
    except (IndexError, KeyError, ValueError):
        return
    raise AssertionError("no error was raised")


# ===================================================================================== integration tests
def free_port():
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Server:
    def __init__(self, results):
        self.port = free_port()
        env = dict(os.environ, RESULTS_DIR=results)
        self.t0 = time.time()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", os.path.join(HERE, "app.py"), "--server.headless", "true",
             "--server.port", str(self.port), "--browser.gatherUsageStats", "false", "--", "--results", results],
            cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        import urllib.request
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://localhost:{self.port}/_stcore/health", timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("Streamlit server did not start")
        self.start_seconds = time.time() - self.t0
        self.url = f"http://localhost:{self.port}"

    def rss_mb(self):
        import psutil
        p = psutil.Process(self.proc.pid)
        return sum(q.memory_info().rss for q in [p] + p.children(recursive=True)) / 1e6

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


DONE_TEXT = "The model predicts"


def run_upload(page, path, timeout=180000):
    """Upload a file and return the seconds until the statement at the end of the result is visible."""
    page.set_input_files('input[type="file"]', [])
    t0 = time.time()
    page.set_input_files('input[type="file"]', path)
    page.get_by_text(DONE_TEXT).first.wait_for(state="visible", timeout=timeout)
    # the two charts are drawn a moment after the text, so the result counts as complete when they are drawn too
    page.wait_for_function("document.querySelectorAll('[data-testid=stVegaLiteChart],[data-testid=stArrowVegaLiteChart]').length >= 2", timeout=timeout)
    return time.time() - t0


def new_page(browser, url):
    page = browser.new_page(viewport={"width": 1400, "height": 2300})
    page.goto(url)
    page.get_by_text("Upload a leaf photo to start.").wait_for(timeout=120000)
    return page


@case("IT01", "integration", "Loading of the saved model, scaling limits, weather profiles and background set from the results folder", "Bundle contains the model in evaluation mode, 16 profiles and a background of weather vectors")
def it01(c):
    b = proto.load_bundle(c["results"])
    assert not b["model"].training and len(b["prof"]) == 16 and b["bg"].shape[1] == 4 and os.path.exists(b["ckpt"])


@case("IT02", "integration", "Start of the Streamlit server and first page", "Server answers the health check; page shows the title and the prototype warning")
def it02(c):
    page = c["page"]
    assert page.get_by_text("Cassava leaf disease diagnosis (research prototype)").first.is_visible()
    assert page.get_by_text("Research prototype.").first.is_visible()


@case("IT03", "integration", "Upload of one leaf photograph through the browser", "Prediction heading, severity or severity note, heatmap, two charts and the statement appear")
def it03(c):
    page = c["page"]
    run_upload(page, c["files"][0])
    assert page.get_by_text("Prediction:").first.is_visible()
    assert page.locator("img").count() >= 2, "heatmap and uploaded image not both shown"
    n_ch = page.locator('[data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"]').count()
    assert n_ch >= 2, f"{n_ch} charts shown"
    assert page.get_by_text(DONE_TEXT).first.is_visible()
    os.makedirs(c["out"], exist_ok=True)
    page.screenshot(path=os.path.join(c["out"], "app_result.png"), full_page=True)


@case("IT04", "integration", "Agreement between the browser result and a direct call of the diagnosis function for the same image and profile", "Same predicted class and the same probability in the statement")
def it04(c):
    page = c["page"]
    run_upload(page, c["files"][0])
    shown = page.get_by_text(DONE_TEXT).first.inner_text()
    # the page shows the first region and season in the lists
    zone = sorted(c["bundle"]["prof"]["zone"].unique())[0]
    o = proto.diagnose(c["bundle"], Image.open(c["files"][0]), zone, "MAM")
    want = f"{ce.CLASS_FULL[ce.CLASS_NAMES[o['cls']]]} ({ce.CLASS_NAMES[o['cls']]}) with a probability of {o['prob'][o['cls']] * 100:.0f}%"
    assert want in shown, (want, shown[:160])


@case("IT05", "integration", "Change of region and season after an upload", "Page recomputes and still shows a complete result")
def it05(c):
    page = c["page"]
    run_upload(page, c["files"][0])
    page.get_by_text("Region").first.scroll_into_view_if_needed()
    page.locator('[data-testid="stSidebar"] [data-testid="stSelectbox"]').first.click()
    page.get_by_role("option").nth(1).click()
    page.get_by_text(DONE_TEXT).first.wait_for(state="visible", timeout=60000)
    assert page.get_by_text("Prediction:").first.is_visible()


@case("IT06", "integration", "Upload of every sample leaf photograph", "Every upload gives a complete result without an error message")
def it06(c):
    for f in c["files"]:
        page = new_page(c["browser"], c["url"])
        run_upload(page, f)
        assert page.locator('[data-testid="stException"]').count() == 0, f
        page.close()


@case("IT07", "integration", "Upload of a file type that is not allowed (text file)", "File is refused by the uploader and no diagnosis is shown")
def it07(c):
    page = new_page(c["browser"], c["url"])
    p = os.path.join(c["tmp"], "notes.txt")
    open(p, "w").write("not an image")
    page.set_input_files('input[type="file"]', p)
    time.sleep(3)
    assert page.get_by_text(DONE_TEXT).count() == 0
    assert page.locator('[data-testid="stException"]').count() == 0
    page.close()


@case("IT08", "integration", "Upload of a corrupted image file (valid extension, damaged content)", "A clear message is shown and the page does not crash with a stack trace")
def it08(c):
    page = new_page(c["browser"], c["url"])
    p = os.path.join(c["tmp"], "broken.jpg")
    open(p, "wb").write(b"\xff\xd8\xff\xe0 this is not a real jpeg")
    page.set_input_files('input[type="file"]', p)
    time.sleep(4)
    crashed = page.locator('[data-testid="stException"]').count() > 0
    page.close()
    assert not crashed, "the page showed an exception trace"


# ===================================================================================== performance tests
def stats(v):
    v = sorted(v)
    return {"n": len(v), "mean": statistics.mean(v), "sd": statistics.stdev(v) if len(v) > 1 else 0.0,
            "median": statistics.median(v), "min": v[0], "max": v[-1], "p95": v[min(len(v) - 1, int(math.ceil(0.95 * len(v))) - 1)]}


def perf_function(c, n=20):
    """Latency of diagnose() split into its stages, on this machine's CPU."""
    b = c["bundle"]
    names = {"gradcam": ce.gradcam, "shap": ce.exact_shap, "text": ce.explanation_text}
    stage = {k: [] for k in names}
    originals = dict(names)

    def wrap(k):
        def f(*a, **kw):
            t = time.perf_counter()
            r = originals[k](*a, **kw)
            stage[k].append(time.perf_counter() - t)
            return r
        return f
    for k in names:
        setattr(ce, {"gradcam": "gradcam", "shap": "exact_shap", "text": "explanation_text"}[k], wrap(k))
    tot = []
    try:
        for i in range(3):
            proto.diagnose(b, c["pil"][0], c["zone"], c["season"])        # warm-up
        for k in stage:
            stage[k].clear()
        for i in range(n):
            im = c["pil"][i % len(c["pil"])]
            t = time.perf_counter()
            proto.diagnose(b, im, c["zone"], c["season"])
            tot.append(time.perf_counter() - t)
    finally:
        for k, nm in {"gradcam": "gradcam", "shap": "exact_shap", "text": "explanation_text"}.items():
            setattr(ce, nm, originals[k])
    out = {"total": stats(tot)}
    out.update({k: stats(v) for k, v in stage.items()})
    out["other"] = {"mean": out["total"]["mean"] - sum(out[k]["mean"] for k in names)}
    return out


def perf_end_to_end(c, n=10):
    res = []
    for i in range(n):
        page = new_page(c["browser"], c["url"])
        res.append(run_upload(page, c["files"][i % len(c["files"])]))
        page.close()
    return res


def perf_by_size(c):
    out = {}
    for name, sz in (("224 x 224 pixels", (224, 224)), ("800 x 600 pixels", (800, 600)), ("1600 x 1200 pixels", (1600, 1200)), ("4000 x 3000 pixels", (4000, 3000))):
        p = os.path.join(c["tmp"], f"size_{sz[0]}.jpg")
        synth(p, sz, seed=3)
        ts = []
        for _ in range(3):
            page = new_page(c["browser"], c["url"])
            ts.append(run_upload(page, p))
            page.close()
        out[name] = {"mean": statistics.mean(ts), "file_mb": os.path.getsize(p) / 1e6}
    return out


# ===================================================================================== runner
def machine():
    cpu = "unknown"
    try:
        for ln in open("/proc/cpuinfo"):
            if ln.startswith("model name"):
                cpu = ln.split(":", 1)[1].strip()
                break
    except Exception:
        cpu = platform.processor() or cpu
    import psutil
    return {"cpu": cpu, "logical_cores": os.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
            "os": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(), "gpu_used": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--images", default=None)
    ap.add_argument("--out", default="test_report")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--skip-browser", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, "tmp")
    os.makedirs(tmp, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.images, "*.jp*g")) + glob.glob(os.path.join(a.images, "*.png"))) if a.images else []
    synthetic = not files
    if synthetic:
        files = [synth(os.path.join(tmp, f"leaf{i}.jpg"), (800, 600), seed=i) for i in range(5)]
    torch.set_num_threads(os.cpu_count() or 1)
    bundle = proto.load_bundle(a.results)
    ctx = {"bundle": bundle, "results": a.results, "files": files, "pil": [Image.open(f).convert("RGB") for f in files],
           "zone": sorted(bundle["prof"]["zone"].unique())[0], "season": "MAM", "tmp": tmp, "out": a.out}
    report = {"machine": machine(), "synthetic_images": synthetic, "smoke_model": bool(bundle["smoke"]), "n_images": len(files),
              "checkpoint": os.path.basename(os.path.dirname(bundle["ckpt"])), "cases": [], "performance": {}}
    preds = []
    for f, im in zip(files, ctx["pil"]):
        o = proto.diagnose(bundle, im, ctx["zone"], ctx["season"])
        stem = os.path.basename(f)
        lab = stem.split("_")[0] if stem.split("_")[0] in ce.CLASS_NAMES else None
        preds.append({"file": stem, "label": lab, "pred": ce.CLASS_NAMES[o["cls"]], "prob": float(o["prob"][o["cls"]])})
    report["images"] = preds
    server = browser = pw = None
    try:
        if not a.skip_browser:
            from playwright.sync_api import sync_playwright
            pw = sync_playwright().start()
            browser = pw.chromium.launch()
            server = Server(a.results)
            report["performance"]["server_start_seconds"] = server.start_seconds
            ctx.update(browser=browser, url=server.url)
            ctx["page"] = new_page(browser, server.url)
            # first diagnosis includes the one-off loading of the model into the server
            t0 = time.time()
            ctx["page"].set_input_files('input[type="file"]', files[0])
            ctx["page"].get_by_text(DONE_TEXT).first.wait_for(state="visible", timeout=300000)
            report["performance"]["first_request_seconds"] = time.time() - t0
            ctx["page"].screenshot(path=os.path.join(a.out, "app_first_result.png"), full_page=False)
            ctx["page"].close()
            ctx["page"] = new_page(browser, server.url)
            ctx["page"].screenshot(path=os.path.join(a.out, "app_start.png"), full_page=False)
        for cs in CASES:
            if cs["kind"] == "integration" and a.skip_browser and cs["id"] != "IT01":
                continue
            t = time.time()
            try:
                cs["fn"](ctx)
                status, msg = "Passed", ""
            except AssertionError as e:
                status, msg = "Failed", str(e)[:300]
            except Exception as e:
                status, msg = "Error", (repr(e) + " " + traceback.format_exc().splitlines()[-3])[:400]
            report["cases"].append({"id": cs["id"], "kind": cs["kind"], "desc": cs["desc"], "expected": cs["expected"],
                                    "status": status, "message": msg, "seconds": time.time() - t})
            print(f"{cs['id']} {status} ({time.time() - t:.1f}s) {msg}", flush=True)
        report["performance"]["function_level"] = perf_function(ctx, n=max(10, a.runs * 2))
        if server:
            e2e = perf_end_to_end(ctx, a.runs)
            report["performance"]["end_to_end_seconds"] = e2e
            report["performance"]["end_to_end"] = stats(e2e)
            report["performance"]["by_image_size"] = perf_by_size(ctx)
            report["performance"]["server_memory_mb"] = server.rss_mb()
    finally:
        if browser:
            browser.close()
        if pw:
            pw.stop()
        if server:
            server.stop()
    n = len(report["cases"])
    report["summary"] = {k: {"total": sum(1 for r in report["cases"] if r["kind"] == k),
                             "passed": sum(1 for r in report["cases"] if r["kind"] == k and r["status"] == "Passed")}
                         for k in ("unit", "integration")}
    json.dump(report, open(os.path.join(a.out, "test_report.json"), "w"), indent=2)
    print(json.dumps(report["summary"]), "| synthetic images:", synthetic)


if __name__ == "__main__":
    main()
