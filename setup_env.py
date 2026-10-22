import hashlib
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

PROJECT_DIR = r"F:\fall_detection"
TMP_DIR = os.path.join(PROJECT_DIR, ".tmp")
CACHE_DIR = os.path.join(PROJECT_DIR, ".pip_cache")
TORCH_HOME_DIR = os.path.join(PROJECT_DIR, ".torch_cache")
YOLO_CONFIG_DIR = os.path.join(PROJECT_DIR, ".ultralytics")
WEIGHTS_DIR = os.path.join(PROJECT_DIR, "weights")
DATA_VIDEOS_DIR = os.path.join(PROJECT_DIR, "data", "videos")

for d in (TMP_DIR, CACHE_DIR, TORCH_HOME_DIR, YOLO_CONFIG_DIR, WEIGHTS_DIR, DATA_VIDEOS_DIR):
    os.makedirs(d, exist_ok=True)

# Set temporary, cache, and model directories exclusively on F:\fall_detection
os.environ["TEMP"] = TMP_DIR
os.environ["TMP"] = TMP_DIR
os.environ["TMPDIR"] = TMP_DIR
os.environ["PIP_CACHE_DIR"] = CACHE_DIR
os.environ["TORCH_HOME"] = TORCH_HOME_DIR
os.environ["YOLO_CONFIG_DIR"] = YOLO_CONFIG_DIR

print(f"[*] Verified Python Executable: {sys.executable}")
print(f"[*] Staging Directory (TEMP): {TMP_DIR}")
print(f"[*] Pip Cache Directory: {CACHE_DIR}")
print(f"[*] Torch Cache Directory: {TORCH_HOME_DIR}")
print(f"[*] Ultralytics Config Directory: {YOLO_CONFIG_DIR}")
print(f"[*] Model Checkpoints Directory: {WEIGHTS_DIR}")
print(f"[*] Sample Evaluation Videos Directory: {DATA_VIDEOS_DIR}")

OPENMMLAB_WEIGHTS = {
    "mmaction2_joint.pth": (
        "https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth",
        "484a394a50246032fef9694396096902909784e8618ccc845d2859dc2b952fea",
    ),
    "mmaction2_bone.pth": (
        "https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d_20221129-c4b44488.pth",
        "c4b44488123d68e19d742ade26d02ff638a911bb0084fcfa7b54d751957318dd",
    ),
    "mmaction2_jm.pth": (
        "https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d_20221129-f18eb408.pth",
        "f18eb408a42eebced02fc0e0dfeed2c6edb320cc1752abc6df855dff0845fc6d",
    ),
    "mmaction2_bm.pth": (
        "https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d_20221129-99c60e2d.pth",
        "99c60e2d462a3978114193f35b5fce465bfa32de2275325913ab8a394900bfb0",
    ),
    "pyskl_j.pth": (
        "https://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu60_xsub_hrnet/j.pth",
        "b4e476a34f2fe641f111cf8b8b66d08fca163a0b45e3df6b9ebb5c93ac9a71b2",
    ),
    "pyskl_b.pth": (
        "https://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu60_xsub_hrnet/b.pth",
        "e9d7cae55964837e6bd9ba5b00f346655da1937317421d79a8963fa9cf0c7fed",
    ),
    "pyskl_jm.pth": (
        "https://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu60_xsub_hrnet/jm.pth",
        "131738564797c3d1c4fc5f3580e4cda0303cffa16efb5f46f4c924e6d9488615",
    ),
    "pyskl_bm.pth": (
        "https://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu60_xsub_hrnet/bm.pth",
        "0e55b74c7d50410ec64ddb4527be6c83145ced62ed671383df45ccd1a2225ac4",
    ),
}


def run_pip(args):
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--cache-dir",
        CACHE_DIR,
    ] + args
    print(f"\n[>] Running: {' '.join(cmd)}")
    subprocess.check_call(cmd, env=os.environ)


def install_torch():
    print("\n--- Installing PyTorch (CUDA 12.6) ---")
    try:
        run_pip([
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            "https://download.pytorch.org/whl/cu126",
        ])
    except subprocess.CalledProcessError:
        run_pip(["torch", "torchvision", "torchaudio"])


def install_libs():
    print("\n--- Installing Core Pipeline & Data Science Libraries ---")
    run_pip([
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "ipykernel",
        "tqdm",
        "tensorboard",
        "opencv-python>=4.8",
        "pyyaml>=6.0",
        "ultralytics>=8.4.0",
        "pytest>=8.0",
        "pytest-timeout>=2.3",
    ])
    print("\n--- Installing ByteTrack Linear Assignment Solver (lapx / lap) ---")
    try:
        run_pip(["lapx>=0.5.5"])
    except subprocess.CalledProcessError:
        print("[!] lapx wheel build failed; trying lap>=0.5.12 fallback...")
        run_pip(["lap>=0.5.12"])


def install_mmaction():
    """MMAction2 for the ST-GCN / pose transforms (no CUDA ops, no decord, no opencv-contrib)."""
    print("\n--- Installing OpenMMLab MMAction2 (mmengine + mmcv-lite + mmaction2 --no-deps) ---")
    run_pip(["mmengine", "addict", "einops", "rich", "termcolor", "yapf", "regex", "packaging", "importlib-metadata"])
    run_pip(["mmcv-lite==2.1.0"])
    run_pip(["--no-deps", "mmaction2==1.2.0"])
    drn_init = os.path.join(PROJECT_DIR, ".venv", "Lib", "site-packages", "mmaction", "models", "localizers", "__init__.py")
    if os.path.exists(drn_init):
        with open(drn_init, "r", encoding="utf-8") as df:
            dcontent = df.read()
        if "from .drn.drn import DRN" in dcontent and "try:" not in dcontent:
            dcontent = dcontent.replace("from .drn.drn import DRN", "try:\n    from .drn.drn import DRN\nexcept ImportError:\n    DRN = None")
            with open(drn_init, "w", encoding="utf-8") as df:
                df.write(dcontent)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_weights():
    print(f"\n--- Verifying OpenMMLab MMAction2 & PYSKL Checkpoints in {WEIGHTS_DIR} ---")
    for fname, (url, expected_sha) in OPENMMLAB_WEIGHTS.items():
        fpath = Path(WEIGHTS_DIR) / fname
        if fpath.exists():
            digest = sha256_file(fpath)
            if digest == expected_sha:
                print(f"[+] Verified {fname} (SHA-256 OK: {digest[:12]}...)")
                continue
            print(f"[!] SHA-256 mismatch for {fname} ({digest} != {expected_sha}); re-downloading...")
        print(f"[>] Downloading {fname} from {url} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, fpath.open("wb") as out:
            out.write(resp.read())
        digest = sha256_file(fpath)
        if digest != expected_sha:
            raise RuntimeError(f"SHA-256 mismatch after download for {fname}: {digest} != {expected_sha}")
        print(f"[+] Downloaded & verified {fname} ({fpath.stat().st_size} bytes)")

    yolo_path = Path(WEIGHTS_DIR) / "yolo26x-pose.pt"
    if yolo_path.exists() and yolo_path.stat().st_size > 1_000_000:
        print(f"[+] Found existing YOLO pose weights at {yolo_path}")
    else:
        print(f"\n--- Downloading Ultralytics yolo26x-pose.pt to {yolo_path} ---")
        try:
            from ultralytics import YOLO
            os.chdir(WEIGHTS_DIR)
            YOLO(str(yolo_path))
            os.chdir(PROJECT_DIR)
            print(f"[+] Verified {yolo_path}")
        except Exception as e:
            os.chdir(PROJECT_DIR)
            print(f"[!] Note: yolo26x-pose.pt auto-download deferred ({e})")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "all"

    if action in ("torch", "all"):
        install_torch()

    if action in ("mmaction", "all"):
        install_mmaction()
    if action in ("libs", "requirements", "all"):
        install_libs()

    if action in ("weights", "all"):
        download_weights()

    print("\n[+] All requested packages and assets configured inside F:\\fall_detection!")
