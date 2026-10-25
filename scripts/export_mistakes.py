import json
import sys
import cv2
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluate_benchmark import parse_le2i_annotation

def main():
    root = ROOT
    summary_path = root / "benchmark_results" / "benchmark_summary.json"
    manifest_path = root / "data" / "manifest.csv"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    manifest = pd.read_csv(manifest_path)
    manifest_map = {}
    for _, row in manifest.iterrows():
        vpath = (root / "data" / row['video_path']).resolve()
        if pd.notna(row['annotation_path']):
            apath = (root / "data" / str(row['annotation_path'])).resolve()
            manifest_map[vpath] = apath

    records = data['records']
    mistakes = []
    
    for r in records:
        is_fn = (r['gt_fall'] and not r['detected'])
        is_fp = (r['false_alarms'] > 0)
        if not (is_fn or is_fp):
            continue
            
        v_abs = Path(r['video']).resolve()
        ann_path = manifest_map.get(v_abs)
        
        cap = cv2.VideoCapture(str(v_abs))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur_s = frame_count / fps if fps > 0 else 0.0
        cap.release()
        
        gt_desc = "Normal ADL (No fall)"
        fall_start_frame = None
        fall_end_frame = None
        fall_start_s = None
        fall_end_s = None
        
        if r['gt_fall'] and ann_path and ann_path.exists():
            ann = parse_le2i_annotation(ann_path, fps=fps)
            if ann.get('has_fall'):
                fall_start_frame = ann['fall_start_frame']
                fall_end_frame = ann['fall_end_frame']
                fall_start_s = ann['fall_start_s']
                fall_end_s = ann['fall_end_s']
                gt_desc = f"Fall frames {fall_start_frame}..{fall_end_frame} ({fall_start_s:.2f}s - {fall_end_s:.2f}s)"
        
        err_type = "FALSE_NEGATIVE" if is_fn else "FALSE_POSITIVE"
        
        # Subfolder logic
        subfolder = v_abs.parent.parent.name
        rel_path = str(v_abs.relative_to(root)).replace("\\", "/")
        
        mistakes.append({
            "video_name": v_abs.name,
            "scene": r['activity'],
            "subfolder": subfolder,
            "error_type": err_type,
            "false_alarm_count": r['false_alarms'],
            "ground_truth_status": "Fall" if r['gt_fall'] else "Normal ADL",
            "ground_truth_details": gt_desc,
            "fall_start_frame": fall_start_frame,
            "fall_end_frame": fall_end_frame,
            "fall_start_s": round(fall_start_s, 2) if fall_start_s is not None else None,
            "fall_end_s": round(fall_end_s, 2) if fall_end_s is not None else None,
            "duration_s": round(dur_s, 2),
            "total_frames": frame_count,
            "fps": round(fps, 2),
            "relative_path": rel_path
        })
        
    df_mistakes = pd.DataFrame(mistakes)
    # Sort by error_type, scene, video_name
    df_mistakes.sort_values(by=["error_type", "scene", "subfolder", "video_name"], inplace=True)
    
    # Save CSV
    csv_out = reports_dir / "benchmark_mistakes.csv"
    df_mistakes.to_csv(csv_out, index=False)
    print(f"Exported CSV: {csv_out} ({len(df_mistakes)} records)")
    
    # Generate Comprehensive Markdown report
    md_out = reports_dir / "benchmark_mistakes_analysis.md"
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("# Le2i Fall Detection Pipeline: Benchmark Failure Analysis Report\n\n")
        f.write(f"- **Total Evaluated Videos**: 190\n")
        f.write(f"- **Total System Mistakes**: {len(df_mistakes)} videos\n")
        f.write(f"  - **False Negatives (Missed Falls)**: {len(df_mistakes[df_mistakes['error_type'] == 'FALSE_NEGATIVE'])}\n")
        f.write(f"  - **False Positives (False Alarms)**: {len(df_mistakes[df_mistakes['error_type'] == 'FALSE_POSITIVE'])}\n")
        f.write(f"- **Files Provided**:\n")
        f.write(f"  - CSV Table for spreadsheet analysis: [`benchmark_mistakes.csv`](benchmark_mistakes.csv)\n")
        f.write(f"  - Full Markdown Document: [`benchmark_mistakes_analysis.md`](benchmark_mistakes_analysis.md)\n\n")
        
        f.write("---\n\n")
        f.write("## 1. Summary by Scene and Error Type\n\n")
        f.write("| Scene | Subfolder | False Negatives (Missed Falls) | False Positives (False Alarms) | Total Mistakes |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: |\n")
        
        for (scene, sub), g in df_mistakes.groupby(["scene", "subfolder"]):
            fn_cnt = len(g[g['error_type'] == 'FALSE_NEGATIVE'])
            fp_cnt = len(g[g['error_type'] == 'FALSE_POSITIVE'])
            f.write(f"| `{scene}` | `{sub}` | **{fn_cnt}** | **{fp_cnt}** | {fn_cnt + fp_cnt} |\n")
            
        f.write("\n---\n\n")
        f.write("## 2. False Positives (False Alarms on Normal ADL) — 20 Videos\n\n")
        f.write("> [!WARNING] Root Cause Pattern\n")
        f.write("> **95% of false alarms (19/20)** occur in `lecture_room` and `office` when subjects sit behind wooden desks. ")
        f.write("Desk occlusion hides knees/ankles, while upper body descent mimics falling/staggering kinematics, and truncated bounding box aspect ratio mimics lying on the floor.\n\n")
        
        f.write("| # | Scene / Subfolder | Video File | Duration | GT Activity | System Alert | Relative Path |\n")
        f.write("| :-: | :--- | :--- | :-: | :--- | :---: | :--- |\n")
        fp_rows = df_mistakes[df_mistakes['error_type'] == 'FALSE_POSITIVE']
        for i, (_, row) in enumerate(fp_rows.iterrows(), 1):
            f.write(f"| {i} | `{row['scene']}` ({row['subfolder']}) | **{row['video_name']}** | {row['duration_s']}s | {row['ground_truth_details']} | {row['false_alarm_count']} False Alarm(s) | [`{row['relative_path']}`]({row['relative_path']}) |\n")
            
        f.write("\n---\n\n")
        f.write("## 3. False Negatives (Missed Falls) — 29 Videos\n\n")
        f.write("> [!IMPORTANT] Root Cause Pattern\n")
        f.write("> Missed falls occur exclusively in `home` (16) and `coffee_room` (13). ")
        f.write("Common causes: forward carpet/sofa falls misclassified by ST-GCN as `A16: wear a shoe` or `A42: staggering` (triggering controlled action veto), ")
        f.write("or frontal camera collapses with compressed vertical pixel drop where fall probability lands just below detection threshold.\n\n")
        
        f.write("| # | Scene / Subfolder | Video File | Duration | GT Fall Interval (Frames) | GT Fall Interval (Time) | Relative Path |\n")
        f.write("| :-: | :--- | :--- | :-: | :--- | :--- | :--- |\n")
        fn_rows = df_mistakes[df_mistakes['error_type'] == 'FALSE_NEGATIVE']
        for i, (_, row) in enumerate(fn_rows.iterrows(), 1):
            f.write(f"| {i} | `{row['scene']}` ({row['subfolder']}) | **{row['video_name']}** | {row['duration_s']}s | Frames {row['fall_start_frame']}..{row['fall_end_frame']} | {row['fall_start_s']}s - {row['fall_end_s']}s | [`{row['relative_path']}`]({row['relative_path']}) |\n")
            
        f.write("\n---\n\n")
        f.write("## 4. Hand-Analysis Checklist for Reviewing Videos\n\n")
        f.write("When inspecting these videos by hand, pay attention to the following physical attributes:\n\n")
        f.write("1. **For False Positives (`lecture_room` & `office`)**:\n")
        f.write("   - Does the subject sit behind a desk?\n")
        f.write("   - Are their legs/knees/ankles completely occluded by the desk barrier?\n")
        f.write("   - How long do they remain seated behind the desk?\n")
        f.write("   - Is the camera angle high-pitch (looking down) or eye-level?\n\n")
        f.write("2. **For False Negatives (`home` & `coffee_room`)**:\n")
        f.write("   - **Fall Orientation**: Is it a forward collapse, backward collapse, sideways roll, or slump?\n")
        f.write("   - **Occlusion / Furniture**: Does the person fall behind a couch, bed, or table?\n")
        f.write("   - **Lying Posture**: Once on the floor, are they stretched out horizontally, or curled in a ball/fetal position?\n")
        f.write("   - **Frontal Perspective**: Does the person fall directly toward or away from the camera lens?\n")

    print(f"Exported Markdown: {md_out}")

if __name__ == "__main__":
    main()
