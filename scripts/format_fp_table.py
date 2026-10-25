import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

def main():
    reports_dir = ROOT / "reports"
    json_path = reports_dir / "false_positives_predicted_times.json"
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    rows = []
    for item in data:
        vname = item["video_name"]
        scene = item["scene"]
        dur = item["duration_s"]
        evs = item["fall_events"]
        if not evs:
            continue
        ev = evs[0]
        t_start = ev["t_start"]
        t_cand = ev["t_candidate"]
        t_alert = ev["t_alert"]
        fr = ev["frame"]
        
        evd = ev.get("evidence", {})
        top5 = evd.get("ntu60_top5", [])
        top1 = f"{top5[0][0]} ({top5[0][1]*100:.1f}%)" if top5 else ""
        top2 = f"{top5[1][0]} ({top5[1][1]*100:.1f}%)" if len(top5) > 1 else ""
        p_fall = evd.get("tsstg_probs", {}).get("Fall Down", 0.0)
        kin = evd.get("kinematics", {})
        drop = kin.get("drop", 0.0)
        vdown = kin.get("vdown", 0.0)
        aspect = kin.get("aspect", 0.0)
        
        rows.append({
            "Video": vname,
            "Scene": scene,
            "Duration (s)": dur,
            "Fall Time (s)": round(t_cand, 2),
            "Fall Frame": fr,
            "Predicted Interval": f"{t_start:.2f}s - {t_alert:.2f}s",
            "ST-GCN Fall Prob": f"{p_fall*100:.1f}%",
            "Top Action 1": top1,
            "Top Action 2": top2,
            "BBox Aspect": round(aspect, 2),
            "Relative Path": item["relative_path"]
        })
        
    df = pd.DataFrame(rows)
    df.sort_values(by=["Scene", "Video"], inplace=True)
    
    # Save CSV
    csv_out = reports_dir / "false_positives_timestamps.csv"
    df.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")
    
    # Save Markdown
    md_out = reports_dir / "false_positives_timestamps.md"
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("# False Positive Videos: Exact Predicted Fall Timestamps & Frames\n\n")
        f.write("> [!NOTE] Crucial Benchmark Finding\n")
        f.write("> In the original Le2i distribution, `Office` (33 videos) and `Lecture_room` (27 videos) **did not include `Annotation_files`**.\n")
        f.write("> The benchmark script automatically assumed all 60 of these videos were 'Normal ADL (No Falls)'.\n")
        f.write("> As confirmed by your visual inspection, **actual falls occur in these videos**! Therefore, these detections are **True Positives** that were scored as False Positives due to missing ground-truth files.\n\n")
        
        f.write("| # | Scene | Video | Duration | Predicted Fall Time | Frame # | Window (s) | ST-GCN Fall Prob | Top Action 1 | Top Action 2 | Aspect | File Link |\n")
        f.write("| :-: | :--- | :--- | :-: | :-: | :-: | :---: | :-: | :--- | :--- | :-: | :--- |\n")
        
        for i, (_, r) in enumerate(df.iterrows(), 1):
            fpath = r['Relative Path']
            f.write(f"| {i} | `{r['Scene']}` | **{r['Video']}** | {r['Duration (s)']}s | **{r['Fall Time (s)']}s** | **{r['Fall Frame']}** | {r['Predicted Interval']} | **{r['ST-GCN Fall Prob']}** | {r['Top Action 1']} | {r['Top Action 2']} | {r['BBox Aspect']} | [`Open Video`]({fpath}) |\n")
            
    print(f"Saved: {md_out}")

if __name__ == "__main__":
    main()
