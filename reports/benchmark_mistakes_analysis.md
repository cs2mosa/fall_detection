# Le2i Fall Detection Pipeline: Benchmark Failure Analysis Report

- **Total Evaluated Videos**: 190
- **Total System Mistakes**: 49 videos
  - **False Negatives (Missed Falls)**: 29
  - **False Positives (False Alarms)**: 20
- **Files Provided**:
  - CSV Table for spreadsheet analysis: [`benchmark_mistakes.csv`](file:///F:/fall_detection/benchmark_mistakes.csv)
  - Full Markdown Document: [`benchmark_mistakes_analysis.md`](file:///F:/fall_detection/benchmark_mistakes_analysis.md)

---

## 1. Summary by Scene and Error Type

| Scene | Subfolder | False Negatives (Missed Falls) | False Positives (False Alarms) | Total Mistakes |
| :--- | :--- | :---: | :---: | :---: |
| `coffee_room` | `Coffee_room_01` | **8** | **0** | 8 |
| `coffee_room` | `Coffee_room_02` | **5** | **1** | 6 |
| `home` | `Home_01` | **13** | **0** | 13 |
| `home` | `Home_02` | **3** | **0** | 3 |
| `lecture_room` | `Lecture_room` | **0** | **10** | 10 |
| `office` | `Office` | **0** | **9** | 9 |

---

## 2. False Positives (False Alarms on Normal ADL) — 20 Videos

> [!WARNING] Root Cause Pattern
> **95% of false alarms (19/20)** occur in `lecture_room` and `office` when subjects sit behind wooden desks. Desk occlusion hides knees/ankles, while upper body descent mimics falling/staggering kinematics, and truncated bounding box aspect ratio mimics lying on the floor.

| # | Scene / Subfolder | Video File | Duration | GT Activity | System Alert | Relative Path |
| :-: | :--- | :--- | :-: | :--- | :---: | :--- |
| 1 | `coffee_room` (Coffee_room_02) | **video (52).avi** | 8.6s | Normal ADL (No fall) | 1 False Alarm(s) | [`data/videos/le2i/Coffee_room_02/Coffee_room_02/Videos/video (52).avi`](file:///F:/fall_detection/data/videos/le2i/Coffee_room_02/Coffee_room_02/Videos/video (52).avi) |
| 2 | `lecture_room` (Lecture_room) | **video (10).avi** | 19.72s | Normal ADL (No fall) | 1 False Alarm(s) | [`data/videos/le2i/Lecture_room/Lecture room/video (10).avi`](file:///F:/fall_detection/data/videos/le2i/Lecture_room/Lecture room/video (10).avi) |
| 3 | `lecture_room` (Lecture_room) | **video (12).avi** | 27.0s | Normal ADL (No fall) | 1 False Alarm(s) | [`data/videos/le2i/Lecture_room/Lecture room/video (12).avi`](file:///F:/fall_detection/data/videos/le2i/Lecture_room/Lecture room/video (12).avi) |
| 4 | `lecture_room` (Lecture_room) | **video (13).avi** | 16.6s | Normal ADL (No fall) | 1 False Alarm(s) | [`data/videos/le2i/Lecture_room/Lecture room/video (13).avi`](file:///F:/fall_detection/data/videos/le2i/Lecture_room/Lecture room/video (13).avi) |
| 5 | `lecture_room` (Lecture_room) | **video (2).avi** | 20.24s | Normal ADL (No fall) | 1 False Alarm(s) | [`data/videos/le2i/Lecture_room/Lecture room/video (2).avi`](file:///F:/fall_detection/data/videos/le2i/Lecture_room/Lecture room/video (2).avi) |
| 6 | `lecture_room` (Lecture_room) | **video (3).avi** | 24.92s | Normal ADL (No fall) | 1 False Alarm(s) | [`data/videos/le2i/Lecture_room/Lecture room/video (3).avi`](file:///F:/fall_detection/data/videos/le2i/Lecture_room/Lecture room/video (3).avi) |
| 7 | `lecture_room` (Lecture_room) | **video (5).avi** | 15.56s | Normal ADL (No fall) | 1 False Alarm(s) | [`data/videos/le2i/Lecture_room/Lecture room/video (5).avi`](file:///F:/fall_detection/data/videos/le2i/Lecture_room/Lecture room/video (5).avi) |
| 8 | `lecture_room` (Lecture_room) | **video (6).avi** | 11.4s | Normal ADL (No fall) | 1 False Alarm(s) | [`data/videos/le2i/Lecture_room/Lecture room/video (6).avi`](file:///F:/fall_detection/data/videos/le2i/Lecture_room/Lecture room/video (6).avi) |
| 9 | `lecture_room` (Lecture_room) | **video (7).avi** | 13.48s | Normal ADL (No fall) | 1 False Alarm(s) | [`data/videos/le2i/Lecture_room/Lecture room/video (7).avi`](file:///F:/fall_detection/data/videos/le2i/Lecture_room/Lecture room/video (7).avi) |
| 10 | `lecture_room` (Lecture_room) | **video (8).avi** | 28.88s | Normal ADL (No fall) | 1 False Alarm(s) | [`data/videos/le2i/Lecture_room/Lecture room/video (8).avi`](file:///F:/fall_detection/data/videos/le2i/Lecture_room/Lecture room/video (8).avi) |
| 11 | `lecture_room` (Lecture_room) | **video (9).avi** | 37.4s | Normal ADL (No fall) | 1 False Alarm(s) | [`data/videos/le2i/Lecture_room/Lecture room/video (9).avi`](file:///F:/fall_detection/data/videos/le2i/Lecture_room/Lecture room/video (9).avi) |
| 12 | `office` (Office) | **video (12).avi** | 11.92s | Normal ADL (No fall) | 1 False Alarm(s) | [`data/videos/le2i/Office/Office/video (12).avi`](file:///F:/fall_detection/data/videos/le2i/Office/Office/video (12).avi) |
| 13 | `office` (Office) | **video (13).avi** | 31.68s | Normal ADL (No fall) | 1 False Alarm(s) | [`data/videos/le2i/Office/Office/video (13).avi`](file:///F:/fall_detection/data/videos/le2i/Office/Office/video (13).avi) |
| 14 | `office` (Office) | **video (16).avi** | 17.12s | Normal ADL (No fall) | 1 False Alarm(s) | [`data/videos/le2i/Office/Office/video (16).avi`](file:///F:/fall_detection/data/videos/le2i/Office/Office/video (16).avi) |
| 15 | `office` (Office) | **video (17).avi** | 8.68s | Normal ADL (No fall) | 1 False Alarm(s) | [`data/videos/le2i/Office/Office/video (17).avi`](file:///F:/fall_detection/data/videos/le2i/Office/Office/video (17).avi) |
| 16 | `office` (Office) | **video (2).avi** | 10.36s | Normal ADL (No fall) | 1 False Alarm(s) | [`data/videos/le2i/Office/Office/video (2).avi`](file:///F:/fall_detection/data/videos/le2i/Office/Office/video (2).avi) |
| 17 | `office` (Office) | **video (4).avi** | 12.8s | Normal ADL (No fall) | 1 False Alarm(s) | [`data/videos/le2i/Office/Office/video (4).avi`](file:///F:/fall_detection/data/videos/le2i/Office/Office/video (4).avi) |
| 18 | `office` (Office) | **video (5).avi** | 9.52s | Normal ADL (No fall) | 1 False Alarm(s) | [`data/videos/le2i/Office/Office/video (5).avi`](file:///F:/fall_detection/data/videos/le2i/Office/Office/video (5).avi) |
| 19 | `office` (Office) | **video (6).avi** | 16.6s | Normal ADL (No fall) | 1 False Alarm(s) | [`data/videos/le2i/Office/Office/video (6).avi`](file:///F:/fall_detection/data/videos/le2i/Office/Office/video (6).avi) |
| 20 | `office` (Office) | **video (7).avi** | 8.68s | Normal ADL (No fall) | 1 False Alarm(s) | [`data/videos/le2i/Office/Office/video (7).avi`](file:///F:/fall_detection/data/videos/le2i/Office/Office/video (7).avi) |

---

## 3. False Negatives (Missed Falls) — 29 Videos

> [!IMPORTANT] Root Cause Pattern
> Missed falls occur exclusively in `home` (16) and `coffee_room` (13). Common causes: forward carpet/sofa falls misclassified by ST-GCN as `A16: wear a shoe` or `A42: staggering` (triggering controlled action veto), or frontal camera collapses with compressed vertical pixel drop where fall probability lands just below detection threshold.

| # | Scene / Subfolder | Video File | Duration | GT Fall Interval (Frames) | GT Fall Interval (Time) | Relative Path |
| :-: | :--- | :--- | :-: | :--- | :--- | :--- |
| 1 | `coffee_room` (Coffee_room_01) | **video (1).avi** | 6.28s | Frames 48.0..80.0 | 1.92s - 3.2s | [`data/videos/le2i/Coffee_room_01/Coffee_room_01/Videos/video (1).avi`](file:///F:/fall_detection/data/videos/le2i/Coffee_room_01/Coffee_room_01/Videos/video (1).avi) |
| 2 | `coffee_room` (Coffee_room_01) | **video (13).avi** | 9.76s | Frames 96.0..115.0 | 3.84s - 4.6s | [`data/videos/le2i/Coffee_room_01/Coffee_room_01/Videos/video (13).avi`](file:///F:/fall_detection/data/videos/le2i/Coffee_room_01/Coffee_room_01/Videos/video (13).avi) |
| 3 | `coffee_room` (Coffee_room_01) | **video (14).avi** | 7.04s | Frames 52.0..76.0 | 2.08s - 3.04s | [`data/videos/le2i/Coffee_room_01/Coffee_room_01/Videos/video (14).avi`](file:///F:/fall_detection/data/videos/le2i/Coffee_room_01/Coffee_room_01/Videos/video (14).avi) |
| 4 | `coffee_room` (Coffee_room_01) | **video (15).avi** | 5.6s | Frames 48.0..73.0 | 1.92s - 2.92s | [`data/videos/le2i/Coffee_room_01/Coffee_room_01/Videos/video (15).avi`](file:///F:/fall_detection/data/videos/le2i/Coffee_room_01/Coffee_room_01/Videos/video (15).avi) |
| 5 | `coffee_room` (Coffee_room_01) | **video (29).avi** | 13.12s | Frames 297.0..328.0 | 11.88s - 13.12s | [`data/videos/le2i/Coffee_room_01/Coffee_room_01/Videos/video (29).avi`](file:///F:/fall_detection/data/videos/le2i/Coffee_room_01/Coffee_room_01/Videos/video (29).avi) |
| 6 | `coffee_room` (Coffee_room_01) | **video (30).avi** | 5.8s | Frames 80.0..104.0 | 3.2s - 4.16s | [`data/videos/le2i/Coffee_room_01/Coffee_room_01/Videos/video (30).avi`](file:///F:/fall_detection/data/videos/le2i/Coffee_room_01/Coffee_room_01/Videos/video (30).avi) |
| 7 | `coffee_room` (Coffee_room_01) | **video (31).avi** | 10.6s | Frames 158.0..176.0 | 6.32s - 7.04s | [`data/videos/le2i/Coffee_room_01/Coffee_room_01/Videos/video (31).avi`](file:///F:/fall_detection/data/videos/le2i/Coffee_room_01/Coffee_room_01/Videos/video (31).avi) |
| 8 | `coffee_room` (Coffee_room_01) | **video (44).avi** | 5.32s | Frames 54.0..86.0 | 2.16s - 3.44s | [`data/videos/le2i/Coffee_room_01/Coffee_room_01/Videos/video (44).avi`](file:///F:/fall_detection/data/videos/le2i/Coffee_room_01/Coffee_room_01/Videos/video (44).avi) |
| 9 | `coffee_room` (Coffee_room_02) | **video (51).avi** | 8.12s | Frames 76.0..117.0 | 3.04s - 4.68s | [`data/videos/le2i/Coffee_room_02/Coffee_room_02/Videos/video (51).avi`](file:///F:/fall_detection/data/videos/le2i/Coffee_room_02/Coffee_room_02/Videos/video (51).avi) |
| 10 | `coffee_room` (Coffee_room_02) | **video (54).avi** | 10.04s | Frames 157.0..179.0 | 6.28s - 7.16s | [`data/videos/le2i/Coffee_room_02/Coffee_room_02/Videos/video (54).avi`](file:///F:/fall_detection/data/videos/le2i/Coffee_room_02/Coffee_room_02/Videos/video (54).avi) |
| 11 | `coffee_room` (Coffee_room_02) | **video (57).avi** | 22.52s | Frames 447.0..492.0 | 17.88s - 19.68s | [`data/videos/le2i/Coffee_room_02/Coffee_room_02/Videos/video (57).avi`](file:///F:/fall_detection/data/videos/le2i/Coffee_room_02/Coffee_room_02/Videos/video (57).avi) |
| 12 | `coffee_room` (Coffee_room_02) | **video (59).avi** | 10.04s | Frames 134.0..157.0 | 5.36s - 6.28s | [`data/videos/le2i/Coffee_room_02/Coffee_room_02/Videos/video (59).avi`](file:///F:/fall_detection/data/videos/le2i/Coffee_room_02/Coffee_room_02/Videos/video (59).avi) |
| 13 | `coffee_room` (Coffee_room_02) | **video (60).avi** | 11.0s | Frames 129.0..157.0 | 5.16s - 6.28s | [`data/videos/le2i/Coffee_room_02/Coffee_room_02/Videos/video (60).avi`](file:///F:/fall_detection/data/videos/le2i/Coffee_room_02/Coffee_room_02/Videos/video (60).avi) |
| 14 | `home` (Home_01) | **video (10).avi** | 13.0s | Frames 135.0..150.0 | 5.62s - 6.25s | [`data/videos/le2i/Home_01/Home_01/Videos/video (10).avi`](file:///F:/fall_detection/data/videos/le2i/Home_01/Home_01/Videos/video (10).avi) |
| 15 | `home` (Home_01) | **video (11).avi** | 13.0s | Frames 137.0..170.0 | 5.71s - 7.08s | [`data/videos/le2i/Home_01/Home_01/Videos/video (11).avi`](file:///F:/fall_detection/data/videos/le2i/Home_01/Home_01/Videos/video (11).avi) |
| 16 | `home` (Home_01) | **video (12).avi** | 9.0s | Frames 161.0..173.0 | 6.71s - 7.21s | [`data/videos/le2i/Home_01/Home_01/Videos/video (12).avi`](file:///F:/fall_detection/data/videos/le2i/Home_01/Home_01/Videos/video (12).avi) |
| 17 | `home` (Home_01) | **video (13).avi** | 10.0s | Frames 156.0..167.0 | 6.5s - 6.96s | [`data/videos/le2i/Home_01/Home_01/Videos/video (13).avi`](file:///F:/fall_detection/data/videos/le2i/Home_01/Home_01/Videos/video (13).avi) |
| 18 | `home` (Home_01) | **video (16).avi** | 11.0s | Frames 186.0..199.0 | 7.75s - 8.29s | [`data/videos/le2i/Home_01/Home_01/Videos/video (16).avi`](file:///F:/fall_detection/data/videos/le2i/Home_01/Home_01/Videos/video (16).avi) |
| 19 | `home` (Home_01) | **video (17).avi** | 10.0s | Frames 154.0..169.0 | 6.42s - 7.04s | [`data/videos/le2i/Home_01/Home_01/Videos/video (17).avi`](file:///F:/fall_detection/data/videos/le2i/Home_01/Home_01/Videos/video (17).avi) |
| 20 | `home` (Home_01) | **video (18).avi** | 9.0s | Frames 135.0..149.0 | 5.62s - 6.21s | [`data/videos/le2i/Home_01/Home_01/Videos/video (18).avi`](file:///F:/fall_detection/data/videos/le2i/Home_01/Home_01/Videos/video (18).avi) |
| 21 | `home` (Home_01) | **video (23).avi** | 10.0s | Frames 165.0..173.0 | 6.87s - 7.21s | [`data/videos/le2i/Home_01/Home_01/Videos/video (23).avi`](file:///F:/fall_detection/data/videos/le2i/Home_01/Home_01/Videos/video (23).avi) |
| 22 | `home` (Home_01) | **video (30).avi** | 8.0s | Frames 123.0..138.0 | 5.12s - 5.75s | [`data/videos/le2i/Home_01/Home_01/Videos/video (30).avi`](file:///F:/fall_detection/data/videos/le2i/Home_01/Home_01/Videos/video (30).avi) |
| 23 | `home` (Home_01) | **video (5).avi** | 8.0s | Frames 109.0..125.0 | 4.54s - 5.21s | [`data/videos/le2i/Home_01/Home_01/Videos/video (5).avi`](file:///F:/fall_detection/data/videos/le2i/Home_01/Home_01/Videos/video (5).avi) |
| 24 | `home` (Home_01) | **video (7).avi** | 9.0s | Frames 139.0..154.0 | 5.79s - 6.42s | [`data/videos/le2i/Home_01/Home_01/Videos/video (7).avi`](file:///F:/fall_detection/data/videos/le2i/Home_01/Home_01/Videos/video (7).avi) |
| 25 | `home` (Home_01) | **video (8).avi** | 11.87s | Frames 125.0..139.0 | 5.21s - 5.79s | [`data/videos/le2i/Home_01/Home_01/Videos/video (8).avi`](file:///F:/fall_detection/data/videos/le2i/Home_01/Home_01/Videos/video (8).avi) |
| 26 | `home` (Home_01) | **video (9).avi** | 14.0s | Frames 149.0..163.0 | 6.21s - 6.79s | [`data/videos/le2i/Home_01/Home_01/Videos/video (9).avi`](file:///F:/fall_detection/data/videos/le2i/Home_01/Home_01/Videos/video (9).avi) |
| 27 | `home` (Home_02) | **video (33).avi** | 8.0s | Frames 127.0..140.0 | 5.29s - 5.83s | [`data/videos/le2i/Home_02/Home_02/Videos/video (33).avi`](file:///F:/fall_detection/data/videos/le2i/Home_02/Home_02/Videos/video (33).avi) |
| 28 | `home` (Home_02) | **video (35).avi** | 9.0s | Frames 136.0..149.0 | 5.67s - 6.21s | [`data/videos/le2i/Home_02/Home_02/Videos/video (35).avi`](file:///F:/fall_detection/data/videos/le2i/Home_02/Home_02/Videos/video (35).avi) |
| 29 | `home` (Home_02) | **video (36).avi** | 8.0s | Frames 124.0..139.0 | 5.17s - 5.79s | [`data/videos/le2i/Home_02/Home_02/Videos/video (36).avi`](file:///F:/fall_detection/data/videos/le2i/Home_02/Home_02/Videos/video (36).avi) |

---

## 4. Hand-Analysis Checklist for Reviewing Videos

When inspecting these videos by hand, pay attention to the following physical attributes:

1. **For False Positives (`lecture_room` & `office`)**:
   - Does the subject sit behind a desk?
   - Are their legs/knees/ankles completely occluded by the desk barrier?
   - How long do they remain seated behind the desk?
   - Is the camera angle high-pitch (looking down) or eye-level?

2. **For False Negatives (`home` & `coffee_room`)**:
   - **Fall Orientation**: Is it a forward collapse, backward collapse, sideways roll, or slump?
   - **Occlusion / Furniture**: Does the person fall behind a couch, bed, or table?
   - **Lying Posture**: Once on the floor, are they stretched out horizontally, or curled in a ball/fetal position?
   - **Frontal Perspective**: Does the person fall directly toward or away from the camera lens?
