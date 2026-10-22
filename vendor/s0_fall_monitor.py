#!/usr/bin/env python3
"""
fall_monitor.py - pose-based FALL / LYING *candidate* detector for a video file, webcam or RTSP stream.

  Ultralytics YOLO-pose + built-in ByteTrack  ->  per-person keypoint features  ->  rule-based candidate events

Thresholds below are the ones tuned on the Le2i dev split in the notebook (yolo26x-pose + ByteTrack).
Alerts are CANDIDATES for operator review. This script never confirms a fall by itself.

Usage
  python fall_monitor.py --source clip.mp4 --show
  python fall_monitor.py --source clip.mp4 --save annotated.mp4 --events events.jsonl
  python fall_monitor.py --source 0                                  # webcam index
  python fall_monitor.py --source rtsp://user:pass@192.168.1.5/stream
  python fall_monitor.py --source clip.mp4 --config thresholds_tuned_yolo26x_pose_pt_bytetrack.yaml   # notebook's saved YAML

Le2i tip: its .avi files carry a broken mp3 track that can crash OpenCV's ffmpeg. Strip it once:
  ffmpeg -i in.avi -an -c:v libx264 -pix_fmt yuv420p out.mp4
Requires: pip install ultralytics lapx pandas pyyaml opencv-python   (lapx provides `lap`, needed by ByteTrack)
"""
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, yaml

# ----------------------------------------------------------------------------------------------------------------------
# Tuned thresholds (notebook: yolo26x-pose.pt@bytetrack, tuned on dev). Everything not tuned keeps its default.
# Motion is in body-heights per second; *_rel values are fractions of the person's own body height.
THRESHOLDS = dict(
    sampling=dict(conf_kp=0.3, smooth_s=0.4),
    fall=dict(drop=0.10, vdown=0.15, window_s=2.0, post_s=3.0, low_hip=0.45, angle=55, aspect=0.9, head_low=0.0,
              stable_s=0.6, max_motion=1.5, hold_s=1.0),
    lying=dict(angle_in=35, angle_out=20, aspect_in=0.5, aspect_out=0.3, T_lying=2.0, hold_s=1.0, max_motion=1.0),
    events=dict(cooldown_s=5.0),
)

def load_cfg(path=None):
    """Embedded thresholds, optionally overridden key-by-key by a YAML file (e.g. the notebook's thresholds_tuned_*.yaml)."""
    cfg = copy.deepcopy(THRESHOLDS)
    if path:
        for sec, vals in (yaml.safe_load(Path(path).read_text()) or {}).items():
            if isinstance(vals, dict) and sec in cfg: cfg[sec].update(vals)
    return cfg

CFG = load_cfg()      # default used by the feature/rule functions below

# ----------------------------------------------------------------------------------------------------------------------
# Features + rules: copied unchanged from the notebook so that streaming behaves like the evaluated pipeline.
def _tidx(t): return pd.to_datetime(t, unit='s')
def _roll(x, t, win_s, fn='mean'):
    r = pd.Series(x, index=_tidx(t)).rolling(f'{max(int(win_s * 1000), 1)}ms', min_periods=1)
    return getattr(r, fn)().to_numpy()

def compute_features(tr, cfg=CFG):
    """Per-track features. angle: torso vs vertical (0=upright, 90=horizontal). hip_rel: hip height above the box
    bottom in units of the person's own (recent 90th-pct) body height. motion: keypoint speed in body-heights/s."""
    s = cfg['sampling']; t = tr['t']; kp = tr['kp']; box = tr['box']
    xy = kp[..., :2].copy(); vis = kp[..., 2] >= s['conf_kp']; xy[~vis] = np.nan
    sh = np.nanmean(xy[:, [5, 6]], axis=1); hp = np.nanmean(xy[:, [11, 12]], axis=1); head = np.nanmean(xy[:, [0, 1, 2]], axis=1)
    d = sh - hp
    angle = np.degrees(np.arctan2(np.abs(d[:, 0]), np.abs(d[:, 1])))
    w = box[:, 2] - box[:, 0]; h = np.maximum(box[:, 3] - box[:, 1], 1.0)
    ref_h = pd.Series(h, index=_tidx(t)).rolling('10000ms', min_periods=1).quantile(0.9).to_numpy()
    hip_rel = (box[:, 3] - hp[:, 1]) / ref_h
    sm = s['smooth_s']
    f = dict(id=tr['id'], t=t, angle=_roll(angle, t, sm), aspect=_roll(w / h, t, sm), hip_rel=_roll(hip_rel, t, sm),
             head_rel=_roll((box[:, 3] - head[:, 1]) / ref_h, t, sm), visibility=vis.mean(axis=1))
    dt = np.maximum(np.diff(t), 1e-3)
    v = np.full(len(t), np.nan); v[1:] = -np.diff(f['hip_rel']) / dt            # downward = positive
    mo = np.full(len(t), np.nan)
    step = np.linalg.norm(xy[1:] - xy[:-1], axis=2)
    mo[1:] = np.nanmean(step, axis=1) / ref_h[1:] / dt
    mo[1:][np.diff(t) > cfg['lying']['hold_s']] = np.nan                         # do not treat tracking gaps as stillness
    f['vdown'] = v; f['motion'] = _roll(mo, t, 1.0)
    return f

def hold_runs(t, cond, hold_s):
    """cond: 1 = condition met, 0 = clearly not met, NaN = unknown (hold state up to hold_s). Returns (start, end) runs."""
    runs, start, last_true, prev = [], None, None, None
    for ti, c in zip(t, cond):
        if start is not None and prev is not None and ti - prev > hold_s:
            runs.append((start, last_true)); start = None
        if c == 1:
            if start is None: start = ti
            last_true = ti
        elif c == 0:
            if start is not None: runs.append((start, last_true)); start = None
        elif start is not None and ti - last_true > hold_s:
            runs.append((start, last_true)); start = None
        prev = ti
    if start is not None: runs.append((start, last_true))
    return runs

def merge_runs(runs, gap):
    out = []
    for s, e in runs:
        if out and s - out[-1][1] <= gap: out[-1] = (out[-1][0], e)
        else: out.append((s, e))
    return out

def _mean_in(x, t, s, e):
    m = (t >= s) & (t <= e); v = x[m]
    return float(np.nanmean(v)) if m.any() and np.isfinite(v).any() else 0.0

def lying_cond(f, p):
    ang, asp = f['angle'], f['aspect']
    ein = np.where(np.isnan(ang), asp > p['aspect_in'] * 1.3, (ang > p['angle_in']) & (asp > p['aspect_in']))   # box-only fallback
    eout = np.where(np.isnan(ang), asp < p['aspect_out'], (ang < p['angle_out']) | (asp < p['aspect_out']))
    return np.where(ein, 1.0, np.where(eout, 0.0, np.nan))      # NaN band = hysteresis

def detect_lying(f, cfg):
    p = cfg['lying']; t = f['t']
    runs = merge_runs(hold_runs(t, lying_cond(f, p), p['hold_s']), cfg['events']['cooldown_s'])
    ev = []
    for s, e in runs:
        if e - s >= p['T_lying'] and _mean_in(f['motion'], t, s, e) <= p['max_motion']:   # floor work has ongoing motion
            ev.append(dict(type='lying', track=f['id'], t_start=float(s), t_candidate=float(s + p['T_lying']), priority='normal'))
    return ev

def detect_falls(f, cfg):
    p = cfg['fall']; t = f['t']; hr = f['hip_rel']
    if len(t) < 3: return []
    win = f"{int(p['window_s'] * 1000)}ms"; idx = _tidx(t)
    drop = (pd.Series(hr, index=idx).rolling(win, min_periods=2).max() - pd.Series(hr, index=idx)).to_numpy()
    vpk = pd.Series(f['vdown'], index=idx).rolling(win, min_periods=1).max().to_numpy()
    hit = (drop > p['drop']) & (vpk > p['vdown'])          # sitting/bending: slow or shallow -> never fires
    ev, last = [], -1e9
    for i in np.flatnonzero(hit):
        if t[i] - last < cfg['events']['cooldown_s']: continue
        sel = (t >= t[i]) & (t <= t[i] + p['post_s'])
        ts = t[sel]
        low = hr[sel] < p['low_hip']                         # hips must end near the floor (rejects sit/kneel/bend)
        # lying-like end state: torso horizontal, wide box, OR head near the floor (falls toward/away from the camera look upright)
        horiz = (f['angle'][sel] > p['angle']) | (f['aspect'][sel] > p['aspect']) | (f['head_rel'][sel] < p.get('head_low', 0.0))
        cond = np.where(low & horiz, 1.0, np.where(np.isnan(hr[sel]), np.nan, 0.0))   # missing pose: hold, never infer
        for s, e in hold_runs(ts, cond, p['hold_s']):
            if e - s >= p['stable_s'] and _mean_in(f['motion'], t, s, e) <= p['max_motion']:   # floor work: ongoing motion
                w = (t >= t[i] - p['window_s']) & (t <= t[i])
                onset = t[w][np.nanargmax(hr[w])] if np.isfinite(hr[w]).any() else t[i]
                ev.append(dict(type='fall', track=f['id'], t_start=float(onset), t_candidate=float(s + p['stable_s']), priority='high'))
                last = t[i]; break
    return ev

# ----------------------------------------------------------------------------------------------------------------------
# Streaming layer: keep a sliding buffer per tracked person and re-run the same detectors on it.
class FallMonitor:
    """update(t, ids, boxes, keypoints) -> list of NEW candidate events (each at most once per `realert_s` per type and person)."""
    def __init__(self, cfg, buffer_s=30.0, eval_every_s=0.5, realert_s=60.0):
        self.cfg, self.buffer_s, self.every, self.realert_s = cfg, buffer_s, eval_every_s, realert_s
        self.tracks, self.last_eval, self.emitted, self.last_cand = {}, {}, {}, {}    # emitted/last_cand: (type, track) -> time of last alert / its t_candidate

    def update(self, t, ids, boxes, kps):
        for tid, b, k in zip(ids, boxes, kps):
            d = self.tracks.setdefault(int(tid), dict(t=[], kp=[], box=[]))
            d['t'].append(float(t)); d['kp'].append(k); d['box'].append(b)
            while d['t'][0] < t - self.buffer_s:
                for key in d: d[key].pop(0)
        for tid in [i for i, d in self.tracks.items() if t - d['t'][-1] > self.buffer_s]:       # forget people not seen for a long time
            del self.tracks[tid]; self.last_eval.pop(tid, None)
        new = []
        for tid in dict.fromkeys(int(i) for i in ids):
            d = self.tracks[tid]
            if len(d['t']) < 5 or t - self.last_eval.get(tid, -1e9) < self.every: continue
            self.last_eval[tid] = t
            f = compute_features(dict(id=tid, t=np.array(d['t']), kp=np.array(d['kp']), box=np.array(d['box'])), self.cfg)
            for ev in detect_falls(f, self.cfg) + detect_lying(f, self.cfg):
                key = (ev['type'], tid)
                if ev['t_candidate'] > t + 1e-6 or t - self.emitted.get(key, -1e9) < self.realert_s: continue
                if ev['t_candidate'] <= self.last_cand.get(key, -1e9) + self.cfg['events']['cooldown_s']: continue   # same old event still in the buffer
                self.emitted[key] = t; self.last_cand[key] = ev['t_candidate']; new.append(dict(ev, t_alert=float(t)))
        return new

# ----------------------------------------------------------------------------------------------------------------------
COCO_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
BANNER = {'fall': ('FALL CANDIDATE - operator review', (0, 0, 220)), 'lying': ('LYING CANDIDATE - operator review', (0, 140, 255))}

def draw(frame, ids, boxes, kps, t, active, conf):
    import cv2
    for tid, b, k in zip(ids, boxes, kps):
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 1); cv2.putText(frame, f'id {int(tid)}', (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        for a, c in COCO_EDGES:
            if k[a, 2] >= conf and k[c, 2] >= conf: cv2.line(frame, tuple(int(v) for v in k[a, :2]), tuple(int(v) for v in k[c, :2]), (0, 255, 0), 2)
    label, col = 'NORMAL', (0, 140, 0)
    for ty in ('lying', 'fall'):                                           # fall banner wins when both are active
        if ty in active and t - active[ty] <= 5.0: label, col = BANNER[ty]
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), col, -1)
    cv2.putText(frame, f'{label}   t={t:5.1f}s', (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def is_live(src): return str(src).isdigit() or str(src).lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

def run(a):
    import cv2
    from ultralytics import YOLO
    cfg = load_cfg(a.config); live = is_live(a.source); src = int(a.source) if str(a.source).isdigit() else a.source
    fps = None
    if not live:
        cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
        fps = fps if fps and 1 <= fps <= 120 else 25.0
    model = YOLO(a.model); mon = FallMonitor(cfg, realert_s=a.realert)
    kw = dict(source=src, stream=True, persist=True, tracker=a.tracker, vid_stride=a.stride, imgsz=a.imgsz, conf=a.conf_det, max_det=10, verbose=False)
    if a.device is not None: kw['device'] = a.device
    sink = open(a.events, 'a') if a.events else None
    writer, active, t0, n_alerts, last_t = None, {}, time.monotonic(), 0, 0.0
    try:
        for n, r in enumerate(model.track(**kw)):
            t = (time.monotonic() - t0) if live else n * a.stride / fps         # file: video time; live stream: wall clock (frames get dropped when slow)
            last_t = t
            have = r.boxes is not None and len(r.boxes) > 0 and r.boxes.id is not None and r.keypoints is not None
            ids = r.boxes.id.int().cpu().tolist() if have else []
            boxes = r.boxes.xyxy.cpu().numpy() if have else []
            kps = r.keypoints.data.cpu().numpy() if have else []
            for ev in mon.update(t, ids, boxes, kps):
                n_alerts += 1; active[ev['type']] = t
                rec = dict(type=ev['type'], track=ev['track'], t_start=round(ev['t_start'], 2), t_candidate=round(ev['t_candidate'], 2), priority=ev['priority'], source=str(a.source))
                print(f"[t={t:6.1f}s] {ev['type'].upper():5s} CANDIDATE  track={ev['track']}  (started ~{ev['t_start']:.1f}s)  priority={ev['priority']}", flush=True)
                if sink: sink.write(json.dumps(rec) + '\n'); sink.flush()
            if a.show or a.save:
                fr = r.orig_img.copy(); draw(fr, ids, boxes, kps, t, active, cfg['sampling']['conf_kp'])
                if a.save and writer is None:
                    writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*'mp4v'), (fps / a.stride) if fps else 10.0, (fr.shape[1], fr.shape[0]))
                if writer: writer.write(fr)
                if a.show:
                    cv2.imshow('fall_monitor', fr)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
    except KeyboardInterrupt:
        pass
    finally:
        if writer: writer.release()
        if sink: sink.close()
        if a.show: cv2.destroyAllWindows()
    print(f'done: {last_t:.1f}s of video processed, {n_alerts} candidate alert(s)')

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', required=True, help='video file, webcam index (0), or rtsp/http URL')
    p.add_argument('--model', default='yolo26x-pose.pt', help='Ultralytics pose weights (yolo11n-pose.pt is ~4x faster)')
    p.add_argument('--tracker', default='bytetrack.yaml', help='bytetrack.yaml or botsort.yaml')
    p.add_argument('--config', help='optional YAML overriding the embedded thresholds')
    p.add_argument('--stride', type=int, default=2, help='process every Nth frame (the notebook used 2)')
    p.add_argument('--imgsz', type=int, default=640); p.add_argument('--conf-det', type=float, default=0.25, help='person detection confidence')
    p.add_argument('--device', default=None, help="e.g. 0 or 'cpu' (default: auto)")
    p.add_argument('--realert', type=float, default=60.0, help='min seconds between repeat alerts of the same type for the same person')
    p.add_argument('--show', action='store_true', help='live preview window (q to quit)')
    p.add_argument('--save', help='write an annotated mp4'); p.add_argument('--events', help='append alerts as JSON lines to this file')
    run(p.parse_args())

if __name__ == '__main__':
    main()
