"""Test script for MouthROIExtractor — run on the server after installing dlib.

Usage (from project root):
    conda activate avsdger
    python scripts/test_mouth_roi.py

What it checks:
    1. dlib imports correctly
    2. All three model files are found
    3. MouthROIExtractor (dlib backend) extracts [T, 1, 96, 96] float32 tensors
    4. Output matches shape of av-hubert precomputed .npy ground truth
    5. Pixel-level correlation with the ground truth is high (>0.5 expected)
    6. haar backend also still works as fallback
"""

import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

def check(label, ok, detail=""):
    status = PASS if ok else FAIL
    print(f"  {status}  {label}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        sys.exit(1)

print("=" * 60)
print("  MouthROIExtractor — dlib backend test")
print("=" * 60)

# ── 1. dlib import ────────────────────────────────────────────────
print("\n[1] dlib installation")
try:
    import dlib
    check("dlib imported", True, f"version {dlib.__version__}")
except ImportError as e:
    check("dlib imported", False, str(e))

# ── 2. model files ────────────────────────────────────────────────
print("\n[2] Model files")
FILES = {
    "face_predictor" : "checkpoints/shape_predictor_68_face_landmarks.dat",
    "cnn_detector"   : "checkpoints/mmod_human_face_detector.dat",
    "mean_face"      : "av_hubert/avhubert/preparation/data/20words_mean_face.npy",
}
for name, path in FILES.items():
    exists = os.path.isfile(path)
    size   = f"{os.path.getsize(path) // 1024} KB" if exists else "missing"
    check(f"{name}: {path}", exists, size)

# ── 3. MouthROIExtractor init ─────────────────────────────────────
print("\n[3] MouthROIExtractor init (dlib)")
from avsd_ger.frontend.mouth_roi import MouthROIExtractor
extractor = MouthROIExtractor(
    backend="dlib",
    face_predictor_path=FILES["face_predictor"],
    cnn_detector_path=FILES["cnn_detector"],
    mean_face_path=FILES["mean_face"],
)
check("MouthROIExtractor created", True)

# ── 4. Extract from sample videos ────────────────────────────────
print("\n[4] Extraction: shape / dtype / range")
VIDEO_DIR = "datasets/lrs2/main/5535415699068794046"
videos = sorted(glob.glob(f"{VIDEO_DIR}/*.mp4"))
if not videos:
    print(f"  No videos found in {VIDEO_DIR} — skipping extraction test")
else:
    print(f"  {'Video':<12} {'T':>4}  {'shape':>18}  {'min':>6}  {'max':>6}")
    print("  " + "-" * 52)
    for v in videos:
        res = extractor.extract_from_file(v)
        ok = (
            res.shape[1:] == torch.Size([1, 96, 96])
            and res.dtype  == torch.float32
            and float(res.min()) >= 0.0
            and float(res.max()) <= 1.0 + 1e-5
        )
        name = os.path.basename(v)
        print(f"  {name:<12} {res.shape[0]:>4}  {str(tuple(res.shape)):>18}"
              f"  {float(res.min()):>6.3f}  {float(res.max()):>6.3f}  "
              f"{PASS if ok else FAIL}")
        if not ok:
            print(f"    ERROR: shape={res.shape}, dtype={res.dtype}")
            sys.exit(1)

# ── 5. Correlation with av-hubert ground truth ────────────────────
print("\n[5] Pixel correlation vs av-hubert dlib ground truth")
GT_DIR  = "data/utts/5535415699068794046"
gt_npys = sorted(glob.glob(f"{GT_DIR}/*.npy"))
if not gt_npys:
    print(f"  No .npy ground-truth files found in {GT_DIR} — skipping")
else:
    corrs = []
    for npy in gt_npys:
        stem    = os.path.basename(npy).replace("_mouth.npy", "")
        vid     = f"{VIDEO_DIR}/{stem}.mp4"
        if not os.path.isfile(vid):
            continue
        res = extractor.extract_from_file(vid).numpy().ravel()
        gt  = np.load(npy).ravel()
        r   = float(np.corrcoef(res, gt)[0, 1])
        corrs.append(r)
        ok  = r > 0.5
        print(f"  {stem}.mp4  Pearson r = {r:.4f}  {PASS if ok else '(low — check head pose)'}")

    mean_r = float(np.mean(corrs))
    print(f"\n  Mean Pearson r = {mean_r:.4f}  (expected >0.5 for frontal face videos)")
    check("Mean correlation > 0.5", mean_r > 0.5, f"r={mean_r:.4f}")

# ── 6. haar fallback ──────────────────────────────────────────────
print("\n[6] haar fallback still works")
ext_haar = MouthROIExtractor(backend="haar")
if videos:
    res_haar = ext_haar.extract_from_file(videos[0])
    check("haar extraction", res_haar.shape[1:] == torch.Size([1, 96, 96]),
          str(tuple(res_haar.shape)))

# ── Done ──────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  All tests PASSED ✓")
print("=" * 60)
