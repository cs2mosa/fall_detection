import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main():
    json_path = ROOT / "reports" / "false_positives_predicted_times.json"
    with open(json_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    pred_map = {(item['scene'], item['video_name']): item['fall_events'][0]['t_candidate'] for item in preds if item['fall_events']}

    import sys
    input_file = Path(sys.argv[1]) if len(sys.argv) > 1 else (ROOT / "data" / "falls_checked.txt")
    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print("| Video | Scene | User Ground Truth | System Detected Time | Detection Latency (s) | Status |")
    print("| :--- | :--- | :---: | :---: | :---: | :---: |")

    diffs = []
    for l in lines:
        l = l.strip()
        if not l:
            continue
        m_path = re.search(r'"([^"]+)"', l)
        m_sec = re.search(r'fall at ([\d\.]+) second', l)
        if m_path and m_sec:
            p = m_path.group(1)
            vname = p.split("\\")[-1]
            user_s = float(m_sec.group(1))
            sc = "coffee_room" if "Coffee" in p else ("lecture_room" if "Lecture" in p else "office")
            sys_s = pred_map.get((sc, vname))
            if sys_s is not None:
                lat = sys_s - user_s
                diffs.append(lat)
                print(f"| **{vname}** | `{sc}` | {user_s:.1f}s | **{sys_s:.2f}s** | +{lat:.2f}s | **TRUE POSITIVE (Accurate)** |")
            else:
                print(f"| **{vname}** | `{sc}` | {user_s:.1f}s | None | Missed | False Negative |")

    print(f"\nMean detection latency after fall start: {sum(diffs)/len(diffs):.2f}s")
    print(f"Max detection latency: {max(diffs):.2f}s, Min: {min(diffs):.2f}s")

if __name__ == "__main__":
    main()
