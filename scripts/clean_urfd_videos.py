#!/usr/bin/env python3
"""clean_urfd_videos.py - Splits URFD videos vertically to isolate clean RGB frames (320x240).

The raw URFD videos are 640x240:
- Left half [0:320]: Infrared Depth map
- Right half [320:640]: RGB video

This script crops each video to the right half (320x240), discarding the depth map,
and atomically replaces the original file.
"""
import concurrent.futures
import subprocess
import time
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = ROOT / "data" / "videos" / "urfd"


def process_video(video_path: Path) -> tuple[bool, str]:
    tmp_path = video_path.with_name(f"{video_path.stem}_clean_tmp.mp4")
    try:
        # Check current resolution
        cap = cv2.VideoCapture(str(video_path))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if orig_w == 320 and orig_h == 240:
            return True, f"Already cleaned: {video_path.name} (320x240)"

        if orig_w != 640 or orig_h != 240:
            return False, f"Unexpected dimensions for {video_path.name}: {orig_w}x{orig_h}"

        # Crop right half: 320 width, 240 height, x-offset 320, y-offset 0
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video_path),
            "-vf", "crop=320:240:320:0",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            str(tmp_path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            if tmp_path.exists():
                tmp_path.unlink()
            return False, f"FFmpeg error on {video_path.name}: {res.stderr}"

        # Verify cropped video
        cap = cv2.VideoCapture(str(tmp_path))
        new_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        new_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        new_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if new_w != 320 or new_h != 240 or new_cnt == 0:
            if tmp_path.exists():
                tmp_path.unlink()
            return False, f"Verification failed for {video_path.name}: got {new_w}x{new_h}, frames={new_cnt}"

        # Atomically replace
        video_path.unlink()
        tmp_path.rename(video_path)
        return True, f"Cleaned: {video_path.name} ({orig_cnt} frames -> 320x240 RGB)"

    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        return False, f"Exception processing {video_path.name}: {e}"


def main():
    videos = sorted(list(VIDEOS_DIR.glob("*.mp4")))
    print(f"Found {len(videos)} videos in {VIDEOS_DIR} to process.")

    t0 = time.time()
    success_count = 0
    errors = []

    # Process with 8 concurrent workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_video = {executor.submit(process_video, v): v for v in videos}
        for future in concurrent.futures.as_completed(future_to_video):
            ok, msg = future.result()
            if ok:
                success_count += 1
                print(f"  [OK] {msg}")
            else:
                errors.append(msg)
                print(f"  [ERROR] {msg}")

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"Cleaned {success_count}/{len(videos)} videos in {elapsed:.2f} seconds.")
    if errors:
        print(f"Errors ({len(errors)}):")
        for err in errors:
            print(f"  - {err}")
    else:
        print("All videos are now pure 320x240 RGB!")
    print("=" * 60)


if __name__ == "__main__":
    main()
