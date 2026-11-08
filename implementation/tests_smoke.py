"""Sanity tests on the SYNTHETIC fixture (python tests_smoke.py). They check that the code learns and wires up
correctly; they say nothing about cassava."""
import copy, os, sys, numpy as np, torch
import cassava_experiment as ce

os.environ.setdefault("CASSAVA_CACHE", "/tmp/smoke_cache")
cfg = copy.deepcopy(ce.DEFAULTS); cfg.update(img_size=96, smoke=True)
log = lambda *a: None
df, X_np, _ = ce.prepare_images("/tmp/smoke_data", "/tmp/smoke_out", cfg, log)
labels = df["label"].values.astype(int)
part = ce.stratified_split(labels)
dev = torch.device("cpu")
ntr = np.array([(labels[part == "train"] == k).sum() for k in range(5)])
cw = len(labels[part == "train"]) / (5 * ntr)
D = {"dev": dev, "X": torch.from_numpy(np.ascontiguousarray(X_np)), "y": torch.as_tensor(labels),
     "grade": torch.ones(len(df)), "mask": torch.zeros(len(df)), "ids": df["image_id"].values,
     "class_weights": torch.as_tensor(cw, dtype=torch.float32),
     "idx": {p: torch.as_tensor(np.where(part == p)[0]) for p in ["train", "val", "test"]}}
E = torch.rand(len(df), 4)
cfg.update(batch_size=16, lr_stage2=1e-3)
# 1. the model learns: validation macro-F1 on the easy synthetic task rises well above chance
model, _, h, info = ce.train_variant("M1", 0, D, E, cfg, 64, 1e-3, 4, 10, log, "t")
print("best val macro-F1 (chance ~0.2):", round(info["best_val_f1"], 3))
assert info["best_val_f1"] > 0.5, "model does not learn"
# 2. gradients reach the environmental branch, and M0 has no environmental parameters
m1 = ce.Net("M1", 64, 256, 64, pretrained=False); m0 = ce.Net("M0", 64, 256, 64, pretrained=False)
x = torch.rand(4, 3, 96, 96); e = torch.rand(4, 4)
lg, s = m1(x, e); (lg.sum() + s.sum()).backward()
assert m1.env[0].weight.grad is not None and m1.env[0].weight.grad.abs().sum() > 0
assert not hasattr(m0, "env") and m0(x)[0].shape == (4, 5)
# 3. severity output is within [1, 5]
assert float(s.min()) >= 1 and float(s.max()) <= 5
# 4. masked severity loss ignores ungraded images
lg = torch.zeros(4, 5); sh = torch.full((4,), 3.0); y = torch.zeros(4, dtype=torch.long)
l0 = ce.loss_fn(lg, sh, y, torch.ones(4), torch.zeros(4), torch.ones(5), 1.0)[2]
l1 = ce.loss_fn(lg, sh, y, torch.ones(4), torch.ones(4), torch.ones(5), 1.0)[2]
assert float(l0) == 0.0 and abs(float(l1) - 4.0) < 1e-6
# 5. Min-Max scaling uses training minima and maxima only (checked by construction in main); Grad-CAM shape and range
cam, c, p = ce.gradcam(model, ce.normalise(ce.to_float(D["X"][:1])), E[:1])
assert cam.shape == (96, 96) and 0 <= cam.min() and cam.max() <= 1 + 1e-6 and abs(p.sum() - 1) < 1e-5
print("all smoke tests passed")
