#!/usr/bin/env python3
"""download_urfd.py - Downloads UR Fall Detection (URFD) dataset videos and annotations.

Downloads:
- 40 ADL videos (cam0)
- 30 Fall videos (cam0)
- 30 Fall videos (cam1)
- urfall-cam0-adls.csv and urfall-cam0-falls.csv
- Generates per-video ground-truth JSON files with fall and lying intervals.
- Generates data/manifest_urfd.csv compatible with benchmark evaluation.
"""
import concurrent.futures
import csv
import json
import ssl
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "https://fenix.ur.edu.pl/mkepski/ds/data"
DATA_DIR = ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos" / "urfd"
ANN_DIR = DATA_DIR / "urfd" / "annotations"
MANIFEST_PATH = DATA_DIR / "manifest_urfd.csv"


def get_ssl_context():
    return ssl._create_unverified_context()


def download_file(url: str, dest_path: Path, max_retries: int = 3) -> Tuple[bool, str]:
    """Downloads a single file with retries and SSL unverified context."""
    if dest_path.exists() and dest_path.stat().st_size > 5000:
        return True, f"Already exists: {dest_path.name} ({dest_path.stat().st_size} bytes)"

    ctx = get_ssl_context()
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, context=ctx, timeout=25) as resp:
                data = resp.read()
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                dest_path.write_bytes(data)
            return True, f"Downloaded: {dest_path.name} ({len(data)} bytes)"
        except Exception as e:
            if attempt == max_retries:
                return False, f"Failed {dest_path.name} after {max_retries} attempts: {e}"
            time.sleep(1.0 * attempt)
    return False, f"Unknown error downloading {dest_path.name}"


def verify_video(video_path: Path) -> Tuple[bool, int, float, int, int]:
    """Verifies that an MP4 video can be decoded and extracts its metadata."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, 0, 0.0, 0, 0
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, frame = cap.read()
    cap.release()
    is_valid = bool(ret and cnt > 0 and w > 0 and h > 0)
    return is_valid, cnt, fps, w, h


def main():
    print("=" * 70)
    print("UR Fall Detection (URFD) Dataset Downloader & Integrator")
    print("=" * 70)

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    ANN_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download official CSV annotations
    print("\n[Step 1/4] Ensuring official annotation CSVs are downloaded...")
    ann_files = ["urfall-cam0-adls.csv", "urfall-cam0-falls.csv"]
    for ann_name in ann_files:
        url = f"{BASE_URL}/{ann_name}"
        dest = ANN_DIR / ann_name
        success, msg = download_file(url, dest)
        print(f"  {msg}")
        if not success:
            print(f"FATAL: Could not download {ann_name}")
            sys.exit(1)

    # 2. Build list of videos to download
    # 40 ADL videos: adl-01 to adl-40 (cam0)
    # 30 Fall videos: fall-01 to fall-30 (cam0 + cam1)
    video_tasks: List[Tuple[str, Path]] = []

    for i in range(1, 41):
        vname = f"adl-{i:02d}-cam0.mp4"
        video_tasks.append((f"{BASE_URL}/{vname}", VIDEOS_DIR / vname))

    for i in range(1, 31):
        vname_c0 = f"fall-{i:02d}-cam0.mp4"
        vname_c1 = f"fall-{i:02d}-cam1.mp4"
        video_tasks.append((f"{BASE_URL}/{vname_c0}", VIDEOS_DIR / vname_c0))
        video_tasks.append((f"{BASE_URL}/{vname_c1}", VIDEOS_DIR / vname_c1))

    print(f"\n[Step 2/4] Downloading {len(video_tasks)} MP4 videos concurrently (6 threads)...")
    download_results = []
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        future_to_task = {
            executor.submit(download_file, url, path): (url, path)
            for url, path in video_tasks
        }
        for future in concurrent.futures.as_completed(future_to_task):
            res = future.result()
            download_results.append(res)
            success, msg = res
            status = "OK" if success else "ERR"
            print(f"  [{status}] {msg}")

    elapsed = time.time() - t0
    failed = [r for r in download_results if not r[0]]
    if failed:
        print(f"\nWARNING: {len(failed)} videos failed to download:")
        for _, msg in failed:
            print(f"  - {msg}")
    else:
        print(f"\nAll {len(video_tasks)} videos downloaded successfully in {elapsed:.1f}s.")

    # 3. Verify video files with OpenCV
    print("\n[Step 3/4] Verifying all downloaded MP4 videos with OpenCV...")
    video_meta: Dict[str, dict] = {}
    corrupted = []
    for _, path in video_tasks:
        is_ok, cnt, fps, w, h = verify_video(path)
        if not is_ok:
            corrupted.append(path.name)
            print(f"  [CORRUPT] {path.name}")
        else:
            video_meta[path.name] = {
                "frames": cnt,
                "fps": fps if fps > 0 else 30.0,
                "width": w,
                "height": h,
                "duration_s": cnt / (fps if fps > 0 else 30.0),
                "size_bytes": path.stat().st_size
            }
    if corrupted:
        print(f"ERROR: {len(corrupted)} corrupted videos found!")
        sys.exit(1)
    else:
        print(f"  Successfully verified {len(video_meta)} videos. 0 corrupt.")

    # 4. Parse ground-truth annotations and create JSONs + manifest
    print("\n[Step 4/4] Parsing annotations and generating manifest_urfd.csv...")
    df_adls = pd.read_csv(ANN_DIR / "urfall-cam0-adls.csv", header=None)
    df_falls = pd.read_csv(ANN_DIR / "urfall-cam0-falls.csv", header=None)

    manifest_rows = []
    lying_videos_count = 0
    total_lying_duration_s = 0.0

    # Process ADL videos
    for i in range(1, 41):
        seq_name = f"adl-{i:02d}"
        vname = f"{seq_name}-cam0.mp4"
        vpath = VIDEOS_DIR / vname
        meta = video_meta.get(vname, {"fps": 30.0, "frames": 0, "duration_s": 0.0})
        fps = meta["fps"]

        sub = df_adls[df_adls[0] == seq_name].sort_values(by=1)
        lying_sub = sub[sub[2] == 1]
        has_lying = len(lying_sub) > 0

        lying_intervals_s = []
        lying_frames = []
        if has_lying:
            # Group consecutive frames
            start_f = None
            prev_f = None
            for f_idx in lying_sub[1]:
                if start_f is None:
                    start_f = f_idx
                    prev_f = f_idx
                elif f_idx == prev_f + 1:
                    prev_f = f_idx
                else:
                    lying_frames.append([int(start_f), int(prev_f)])
                    lying_intervals_s.append([round((start_f - 1) / fps, 3), round(prev_f / fps, 3)])
                    start_f = f_idx
                    prev_f = f_idx
            if start_f is not None:
                lying_frames.append([int(start_f), int(prev_f)])
                lying_intervals_s.append([round((start_f - 1) / fps, 3), round(prev_f / fps, 3)])

            lying_videos_count += 1
            dur = sum(e - s for s, e in lying_intervals_s)
            total_lying_duration_s += dur

        gt_json = {
            "sequence": seq_name,
            "camera": "cam0",
            "has_fall": False,
            "fall_start_s": None,
            "fall_end_s": None,
            "fall_frames": None,
            "has_lying": has_lying,
            "lying_intervals_s": lying_intervals_s,
            "lying_frames": lying_frames,
            "fps": fps,
            "total_frames": meta["frames"],
            "duration_s": round(meta["duration_s"], 3)
        }

        ann_json_name = f"{vpath.stem}.json"
        ann_json_path = ANN_DIR / ann_json_name
        ann_json_path.write_text(json.dumps(gt_json, indent=2), encoding="utf-8")

        rel_vpath = f"videos/urfd/{vname}"
        rel_annpath = f"urfd/annotations/{ann_json_name}"
        manifest_rows.append({
            "video_path": rel_vpath,
            "annotation_path": rel_annpath,
            "activity": "urfd_adl"
        })

    # Process Fall videos (cam0 and cam1)
    for i in range(1, 31):
        seq_name = f"fall-{i:02d}"
        sub = df_falls[df_falls[0] == seq_name].sort_values(by=1)

        f0 = sub[sub[2] == 0]  # fall transition
        f1 = sub[sub[2] == 1]  # lying on floor

        has_fall = len(f0) > 0 or len(f1) > 0
        has_lying = len(f1) > 0

        # Cameras cam0 and cam1
        for cam in ["cam0", "cam1"]:
            vname = f"{seq_name}-{cam}.mp4"
            vpath = VIDEOS_DIR / vname
            meta = video_meta.get(vname, {"fps": 30.0, "frames": 0, "duration_s": 0.0})
            fps = meta["fps"]

            fall_start_s = round((f0[1].min() - 1) / fps, 3) if len(f0) > 0 else (round((f1[1].min() - 1) / fps, 3) if has_lying else None)
            fall_end_s = round(f0[1].max() / fps, 3) if len(f0) > 0 else (fall_start_s + 1.0 if fall_start_s else None)
            fall_frames = [int(f0[1].min()), int(f0[1].max())] if len(f0) > 0 else None

            lying_intervals_s = []
            lying_frames = []
            if has_lying:
                start_f = None
                prev_f = None
                for f_idx in f1[1]:
                    if start_f is None:
                        start_f = f_idx
                        prev_f = f_idx
                    elif f_idx == prev_f + 1:
                        prev_f = f_idx
                    else:
                        lying_frames.append([int(start_f), int(prev_f)])
                        lying_intervals_s.append([round((start_f - 1) / fps, 3), round(prev_f / fps, 3)])
                        start_f = f_idx
                        prev_f = f_idx
                if start_f is not None:
                    lying_frames.append([int(start_f), int(prev_f)])
                    lying_intervals_s.append([round((start_f - 1) / fps, 3), round(prev_f / fps, 3)])

                lying_videos_count += 1
                dur = sum(e - s for s, e in lying_intervals_s)
                total_lying_duration_s += dur

            gt_json = {
                "sequence": seq_name,
                "camera": cam,
                "has_fall": True,
                "fall_start_s": fall_start_s,
                "fall_end_s": fall_end_s,
                "fall_frames": fall_frames,
                "has_lying": has_lying,
                "lying_intervals_s": lying_intervals_s,
                "lying_frames": lying_frames,
                "fps": fps,
                "total_frames": meta["frames"],
                "duration_s": round(meta["duration_s"], 3)
            }

            ann_json_name = f"{vpath.stem}.json"
            ann_json_path = ANN_DIR / ann_json_name
            ann_json_path.write_text(json.dumps(gt_json, indent=2), encoding="utf-8")

            rel_vpath = f"videos/urfd/{vname}"
            rel_annpath = f"urfd/annotations/{ann_json_name}"
            manifest_rows.append({
                "video_path": rel_vpath,
                "annotation_path": rel_annpath,
                "activity": "urfd_fall"
            })

    # Write manifest_urfd.csv
    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(MANIFEST_PATH, index=False)
    print(f"  Generated manifest at: {MANIFEST_PATH} with {len(df_manifest)} entries.")

    # Dataset Summary
    total_size_mb = sum(m["size_bytes"] for m in video_meta.values()) / (1024 * 1024)
    total_dur_min = sum(m["duration_s"] for m in video_meta.values()) / 60.0

    print("\n" + "=" * 70)
    print("URFD DATASET INTEGRATION COMPLETE")
    print("=" * 70)
    print(f"  Total Video Files:       {len(video_meta)}")
    print(f"    - ADL Videos (cam0):   40")
    print(f"    - Fall Videos (cam0):  30")
    print(f"    - Fall Videos (cam1):  30")
    print(f"  Total Video Storage:     {total_size_mb:.2f} MB")
    print(f"  Total Video Duration:    {total_dur_min:.2f} minutes")
    print(f"  Videos with Lying Class: {lying_videos_count} (16 ADLs + 60 Falls [cam0+cam1])")
    print(f"  Total Ground-Truth Lying Duration: {total_lying_duration_s:.1f} seconds")
    print(f"  Manifest Path:           {MANIFEST_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
