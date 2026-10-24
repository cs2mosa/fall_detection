import cv2
import json
import sys
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fall_monitor import FallMonitor, CFG

def main():
    root = ROOT
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # 20 FP videos
    fp_vids = [
        "data/videos/le2i/Coffee_room_02/Coffee_room_02/Videos/video (52).avi",
        "data/videos/le2i/Lecture_room/Lecture room/video (2).avi",
        "data/videos/le2i/Lecture_room/Lecture room/video (3).avi",
        "data/videos/le2i/Lecture_room/Lecture room/video (5).avi",
        "data/videos/le2i/Lecture_room/Lecture room/video (6).avi",
        "data/videos/le2i/Lecture_room/Lecture room/video (7).avi",
        "data/videos/le2i/Lecture_room/Lecture room/video (8).avi",
        "data/videos/le2i/Lecture_room/Lecture room/video (9).avi",
        "data/videos/le2i/Lecture_room/Lecture room/video (10).avi",
        "data/videos/le2i/Lecture_room/Lecture room/video (12).avi",
        "data/videos/le2i/Lecture_room/Lecture room/video (13).avi",
        "data/videos/le2i/Office/Office/video (2).avi",
        "data/videos/le2i/Office/Office/video (4).avi",
        "data/videos/le2i/Office/Office/video (5).avi",
        "data/videos/le2i/Office/Office/video (6).avi",
        "data/videos/le2i/Office/Office/video (7).avi",
        "data/videos/le2i/Office/Office/video (12).avi",
        "data/videos/le2i/Office/Office/video (13).avi",
        "data/videos/le2i/Office/Office/video (16).avi",
        "data/videos/le2i/Office/Office/video (17).avi"
    ]
    
    model = YOLO("weights/yolo26x-pose.pt")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    results = []
    
    for vrel in fp_vids:
        vpath = root / vrel
        cap = cv2.VideoCapture(str(vpath))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 240)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 320)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        
        monitor = FallMonitor(cfg=CFG, device=device, img_shape=(h, w))
        
        fall_events = []
        idx = 0
        
        for res in model.track(source=str(vpath), tracker="bytetrack.yaml", conf=0.25, imgsz=640, stream=True, verbose=False, device=device):
            t_curr = float(idx) / fps
            idx += 1
            
            if res.boxes is None or len(res.boxes) == 0:
                evs = monitor.update(t_curr, [], [], [])
            else:
                boxes = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else np.arange(len(boxes), dtype=int)
                kps = res.keypoints.data.cpu().numpy() if res.keypoints is not None else np.zeros((len(boxes), 17, 3), dtype=np.float32)
                evs = monitor.update(t_curr, ids, boxes, kps, frame=getattr(res, "orig_img", None))
                
            for e in evs:
                if e.get("type") == "fall":
                    t_cand = e.get("t_candidate", t_curr)
                    t_alert = e.get("t_alert", t_curr)
                    t_start = e.get("t_start", t_cand)
                    frame_num = int(round(t_cand * fps))
                    fall_events.append({
                        "t_candidate": round(t_cand, 2),
                        "t_alert": round(t_alert, 2),
                        "t_start": round(t_start, 2),
                        "frame": frame_num,
                        "evidence": e.get("evidence", {})
                    })
                    
        scene = "coffee_room" if "Coffee_room" in vrel else ("lecture_room" if "Lecture_room" in vrel else "office")
        results.append({
            "video_name": vpath.name,
            "scene": scene,
            "relative_path": vrel,
            "total_frames": total_frames,
            "fps": fps,
            "duration_s": round(total_frames / fps, 2),
            "fall_events": fall_events
        })
        print(f"Done: {vpath.name} -> {len(fall_events)} fall event(s)")
        
    out_json = reports_dir / "false_positives_predicted_times.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_json}")

if __name__ == "__main__":
    main()
