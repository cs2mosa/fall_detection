#!/usr/bin/env python3
"""create_demos.py - Generates polished annotated demonstration videos for Fall and Lying detection.

Outputs to:
  data/demos/demo_fall.mp4   (Annotated fall sequence)
  data/demos/demo_lying.mp4  (Annotated persistent lying sequence)
"""
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fall_monitor import FallMonitor, load_cfg, compute_features

DEMOS_DIR = Path("F:/fall_detection/data/demos")
DEMOS_DIR.mkdir(parents=True, exist_ok=True)

# COCO 17-Keypoint skeleton connectivity
SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Head
    (5, 6),                                    # Shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),           # Arms
    (5, 11), (6, 12), (11, 12),                # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)     # Legs
]

COLOR_NORMAL = (40, 200, 40)       # Green
COLOR_FALL = (30, 30, 235)         # Red
COLOR_LYING = (0, 150, 255)        # Amber / Orange
COLOR_TEXT = (255, 255, 255)
COLOR_BG_DARK = (20, 20, 20)


def draw_hud(
    canvas: np.ndarray,
    video_title: str,
    t: float,
    status: str,
    action_probs: Dict[str, float],
    kinematics: Dict[str, float],
    status_color: Tuple[int, int, int]
):
    """Draws an operator HUD overlay on the upper portion of the frame."""
    h, w = canvas.shape[:2]
    overlay = canvas.copy()

    # Upper telemetry banner
    cv2.rectangle(overlay, (0, 0), (w, 64), COLOR_BG_DARK, -1)
    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)

    # Title & time
    cv2.putText(canvas, f"AI FALL & LYING MONITOR | {video_title}", (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"T: {t:05.2f}s", (12, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 230, 255), 2, cv2.LINE_AA)

    # Status Pill / Badge
    pill_w, pill_h = 240, 34
    pill_x = w - pill_w - 12
    pill_y = 14
    cv2.rectangle(canvas, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), status_color, -1)
    cv2.rectangle(canvas, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), (255, 255, 255), 1)

    text_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
    tx = pill_x + (pill_w - text_size[0]) // 2
    ty = pill_y + (pill_h + text_size[1]) // 2 - 2
    cv2.putText(canvas, status, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # Telemetry metrics row
    p_fall = action_probs.get("Fall Down", 0.0)
    p_walk = action_probs.get("Walking", 0.0)
    p_sit = action_probs.get("Sit down", 0.0)
    angle = kinematics.get("angle", 0.0)
    aspect = kinematics.get("aspect", 0.0)

    telem_str = f"Fall Prob: {p_fall:4.1%} | Walk: {p_walk:4.1%} | Angle: {angle:4.1f}° | Aspect: {aspect:3.2f}"
    cv2.putText(canvas, telem_str, (120, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 210, 220), 1, cv2.LINE_AA)


def draw_person(
    canvas: np.ndarray,
    box: np.ndarray,
    kps: np.ndarray,
    tid: int,
    state: str,
    state_color: Tuple[int, int, int],
    scale_factor: float
):
    """Draws keypoints, skeleton links, bounding box, and floating tag on top of person."""
    # Scale box and keypoints to canvas resolution
    bx1, by1, bx2, by2 = (box * scale_factor).astype(int)
    scaled_kps = kps.copy()
    scaled_kps[:, :2] *= scale_factor

    # Draw skeleton bones
    for p1_idx, p2_idx in SKELETON_PAIRS:
        c1, c2 = scaled_kps[p1_idx, 2], scaled_kps[p2_idx, 2]
        if c1 >= 0.25 and c2 >= 0.25:
            pt1 = (int(scaled_kps[p1_idx, 0]), int(scaled_kps[p1_idx, 1]))
            pt2 = (int(scaled_kps[p2_idx, 0]), int(scaled_kps[p2_idx, 1]))
            cv2.line(canvas, pt1, pt2, (120, 220, 255), 2, cv2.LINE_AA)

    # Draw keypoint dots
    for j in range(17):
        if scaled_kps[j, 2] >= 0.25:
            pt = (int(scaled_kps[j, 0]), int(scaled_kps[j, 1]))
            cv2.circle(canvas, pt, 4, state_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 5, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw person bounding box
    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), state_color, 2, cv2.LINE_AA)

    # Floating Tag directly on top of person
    tag_h = 24
    tag_w = max(180, (bx2 - bx1))
    tag_x1 = max(4, bx1)
    tag_y1 = max(68, by1 - tag_h - 4)
    tag_x2 = tag_x1 + tag_w
    tag_y2 = tag_y1 + tag_h

    # Badge background
    overlay = canvas.copy()
    cv2.rectangle(overlay, (tag_x1, tag_y1), (tag_x2, tag_y2), state_color, -1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
    cv2.rectangle(canvas, (tag_x1, tag_y1), (tag_x2, tag_y2), (255, 255, 255), 1)

    # Label text
    label_text = f"ID #{tid} | {state}"
    cv2.putText(canvas, label_text, (tag_x1 + 6, tag_y1 + 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)


def render_annotated_demo(
    input_video_path: Path,
    output_mp4_path: Path,
    demo_type: str,  # "fall" or "lying"
    display_title: str,
    upscale_w: int = 640,
    upscale_h: int = 480
):
    print(f"\nProcessing {demo_type.upper()} demo: {input_video_path.name} -> {output_mp4_path.name}")
    cap = cv2.VideoCapture(str(input_video_path))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scale_factor = upscale_w / orig_w
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    raw_tmp_path = output_mp4_path.with_name(f"{output_mp4_path.stem}_raw.mp4")
    out = cv2.VideoWriter(str(raw_tmp_path), fourcc, orig_fps, (upscale_w, upscale_h))

    # Initialize YOLO and FallMonitor
    cfg = load_cfg()
    model = YOLO("weights/yolo26x-pose.pt")
    monitor = FallMonitor(cfg=cfg, img_shape=(orig_h, orig_w), device="cuda:0")

    # Tracking state across frames
    active_fall_alert = False
    active_lying_alert = False
    alert_hold_frames = 0

    idx = 0
    t0 = time.time()
    for res in model.track(
        source=str(input_video_path),
        tracker="bytetrack.yaml",
        conf=0.25,
        stream=True,
        verbose=False,
        device="cuda:0"
    ):
        t = idx / orig_fps
        idx += 1

        orig_frame = res.orig_img
        canvas = cv2.resize(orig_frame, (upscale_w, upscale_h), interpolation=cv2.INTER_LINEAR)

        # Track boxes and keypoints
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0, 4))
        ids = res.boxes.id.cpu().numpy().astype(int) if (res.boxes is not None and res.boxes.id is not None) else np.array([1] * len(boxes))
        kps = res.keypoints.data.cpu().numpy() if res.keypoints is not None else np.zeros((len(boxes), 17, 3), dtype=np.float32)

        # Monitor update
        events = monitor.update(t, ids, boxes, kps, frame=orig_frame)

        # Check for newly emitted events
        for ev in events:
            ev_type = ev.get("type")
            if ev_type == "fall":
                active_fall_alert = True
                active_lying_alert = False
                alert_hold_frames = int(orig_fps * 4.0)  # Hold alert banner
                print(f"  [FRAME {idx:03d} | t={t:5.2f}s] >>> FALL DETECTED (t_cand={ev.get('t_candidate'):.2f}s)")
            elif ev_type == "lying":
                active_lying_alert = True
                alert_hold_frames = int(orig_fps * 4.0)
                print(f"  [FRAME {idx:03d} | t={t:5.2f}s] >>> LYING DETECTED (t_cand={ev.get('t_candidate'):.2f}s)")

        # Decrement alert hold
        if alert_hold_frames > 0:
            alert_hold_frames -= 1
            if alert_hold_frames == 0 and not active_lying_alert:
                active_fall_alert = False

        # Extract telemetry from first track if present
        action_probs: Dict[str, float] = {}
        kinematics: Dict[str, float] = {}

        if len(ids) > 0:
            tid = int(ids[0])
            b = boxes[0]
            k = kps[0]

            # Get track internal features if available
            tr_data = monitor.tracks.get(tid)
            if tr_data and len(tr_data["t"]) >= 3:
                try:
                    f = compute_features(
                        dict(id=tid, t=np.array(tr_data["t"]), kp=np.array(tr_data["kp"]), box=np.array(tr_data["box"])),
                        cfg,
                        ref_scale=tr_data["ref_scale"]
                    )
                    kinematics["angle"] = float(f["angle"][-1]) if np.isfinite(f["angle"][-1]) else 0.0
                    kinematics["aspect"] = float(f["aspect"][-1]) if np.isfinite(f["aspect"][-1]) else 0.0
                except Exception:
                    pass

            if tr_data and tr_data.get("probs_hist"):
                latest_p = tr_data["probs_hist"][-1][1]
                action_probs["Walking"] = float(latest_p[1]) if len(latest_p) > 1 else 0.0
                action_probs["Sit down"] = float(latest_p[5]) if len(latest_p) > 5 else 0.0
                action_probs["Fall Down"] = float(latest_p[6]) if len(latest_p) > 6 else 0.0

            # Determine person state
            if active_fall_alert:
                person_state = "ALERT: FALL DETECTED"
                person_color = COLOR_FALL
            elif active_lying_alert:
                person_state = "ALERT: LYING DETECTED"
                person_color = COLOR_LYING
            else:
                person_state = "NORMAL (Upright)"
                person_color = COLOR_NORMAL

            draw_person(canvas, b, k, tid, person_state, person_color, scale_factor)

        # Determine overall HUD status
        if active_fall_alert:
            hud_status = "CRITICAL: FALL DETECTED"
            hud_color = COLOR_FALL
        elif active_lying_alert:
            hud_status = "MONITOR: LYING ON FLOOR"
            hud_color = COLOR_LYING
        else:
            hud_status = "NORMAL: MONITORING"
            hud_color = COLOR_NORMAL

        draw_hud(canvas, display_title, t, hud_status, action_probs, kinematics, hud_color)
        out.write(canvas)

    cap.release()
    out.release()

    # Re-encode with FFmpeg H.264 for universal compatibility
    print(f"  Re-encoding to H.264: {output_mp4_path.name}...")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(raw_tmp_path),
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        str(output_mp4_path)
    ]
    subprocess.run(cmd, check=True)
    if raw_tmp_path.exists():
        raw_tmp_path.unlink()

    dur = time.time() - t0
    size_mb = output_mp4_path.stat().st_size / (1024 * 1024)
    print(f"  Finished {output_mp4_path.name} ({total_frames} frames, {size_mb:.2f} MB, {dur:.1f}s)")


def main():
    print("=" * 70)
    print("GENERATING DEMONSTRATION VIDEOS")
    print("=" * 70)

    # 1. Fall Demo
    # fall-01-cam0.mp4: Person walks across room, trips, falls, hits floor.
    fall_source = Path("F:/fall_detection/data/videos/urfd/fall-01-cam0.mp4")
    fall_dest = DEMOS_DIR / "demo_fall.mp4"
    render_annotated_demo(
        input_video_path=fall_source,
        output_mp4_path=fall_dest,
        demo_type="fall",
        display_title="Fall Incident Detection Demo",
        upscale_w=640,
        upscale_h=480
    )

    # 2. Lying Demo
    # adl-10-cam0.mp4: Person walks, stands, and intentionally lies down on floor.
    lying_source = Path("F:/fall_detection/data/videos/urfd/adl-10-cam0.mp4")
    lying_dest = DEMOS_DIR / "demo_lying.mp4"
    render_annotated_demo(
        input_video_path=lying_source,
        output_mp4_path=lying_dest,
        demo_type="lying",
        display_title="Persistent Lying Detection Demo",
        upscale_w=640,
        upscale_h=480
    )

    print("\n" + "=" * 70)
    print("DEMONSTRATION VIDEOS PUBLISHED SUCCESSFULLY")
    print("=" * 70)
    print(f"  1. Fall Demo:  {fall_dest} ({fall_dest.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  2. Lying Demo: {lying_dest} ({lying_dest.stat().st_size / (1024*1024):.2f} MB)")
    print("=" * 70)


if __name__ == "__main__":
    main()
