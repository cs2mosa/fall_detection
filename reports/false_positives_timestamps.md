# False Positive Videos: Exact Predicted Fall Timestamps & Frames

> [!NOTE] Crucial Benchmark Finding
> In the original Le2i distribution, `Office` (33 videos) and `Lecture_room` (27 videos) **did not include `Annotation_files`**.
> The benchmark script automatically assumed all 60 of these videos were 'Normal ADL (No Falls)'.
> As confirmed by your visual inspection, **actual falls occur in these videos**! Therefore, these detections are **True Positives** that were scored as False Positives due to missing ground-truth files.

| # | Scene | Video | Duration | Predicted Fall Time | Frame # | Window (s) | ST-GCN Fall Prob | Top Action 1 | Top Action 2 | Aspect | File Link |
| :-: | :--- | :--- | :-: | :-: | :-: | :---: | :-: | :--- | :--- | :-: | :--- |
| 1 | `coffee_room` | **video (52).avi** | 8.6s | **4.64s** | **116** | 3.08s - 4.76s | **30.7%** | staggering (52.3%) | falling (30.7%) | 2.47 | [`Open Video`](file:///F:/fall_detection/data/videos/le2i/Coffee_room_02/Coffee_room_02/Videos/video (52).avi) |
| 2 | `lecture_room` | **video (10).avi** | 19.72s | **15.92s** | **398** | 13.36s - 16.72s | **4.2%** | wear a shoe (47.8%) | take off a shoe (20.4%) | 1.16 | [`Open Video`](file:///F:/fall_detection/data/videos/le2i/Lecture_room/Lecture room/video (10).avi) |
| 3 | `lecture_room` | **video (12).avi** | 27.0s | **25.24s** | **631** | 22.48s - 25.48s | **20.9%** | staggering (21.8%) | falling (20.9%) | 0.97 | [`Open Video`](file:///F:/fall_detection/data/videos/le2i/Lecture_room/Lecture room/video (12).avi) |
| 4 | `lecture_room` | **video (13).avi** | 16.6s | **13.56s** | **339** | 11.28s - 13.92s | **56.6%** | falling (56.6%) | wear a shoe (18.5%) | 1.84 | [`Open Video`](file:///F:/fall_detection/data/videos/le2i/Lecture_room/Lecture room/video (13).avi) |
| 5 | `lecture_room` | **video (2).avi** | 20.24s | **16.52s** | **413** | 14.52s - 16.56s | **40.8%** | staggering (51.8%) | falling (40.8%) | 1.47 | [`Open Video`](file:///F:/fall_detection/data/videos/le2i/Lecture_room/Lecture room/video (2).avi) |
| 6 | `lecture_room` | **video (3).avi** | 24.92s | **20.08s** | **502** | 18.00s - 20.32s | **24.5%** | staggering (59.2%) | falling (24.5%) | 1.5 | [`Open Video`](file:///F:/fall_detection/data/videos/le2i/Lecture_room/Lecture room/video (3).avi) |
| 7 | `lecture_room` | **video (5).avi** | 15.56s | **11.72s** | **293** | 9.52s - 12.16s | **40.2%** | falling (40.2%) | wear a shoe (21.7%) | 2.1 | [`Open Video`](file:///F:/fall_detection/data/videos/le2i/Lecture_room/Lecture room/video (5).avi) |
| 8 | `lecture_room` | **video (6).avi** | 11.4s | **7.88s** | **197** | 5.24s - 8.24s | **86.3%** | falling (86.3%) | wear a shoe (5.8%) | 2.33 | [`Open Video`](file:///F:/fall_detection/data/videos/le2i/Lecture_room/Lecture room/video (6).avi) |
| 9 | `lecture_room` | **video (7).avi** | 13.48s | **10.36s** | **259** | 7.60s - 10.60s | **73.7%** | falling (73.7%) | staggering (14.2%) | 1.7 | [`Open Video`](file:///F:/fall_detection/data/videos/le2i/Lecture_room/Lecture room/video (7).avi) |
| 10 | `lecture_room` | **video (8).avi** | 28.88s | **24.88s** | **622** | 22.52s - 25.28s | **76.8%** | falling (76.8%) | staggering (8.5%) | 2.17 | [`Open Video`](file:///F:/fall_detection/data/videos/le2i/Lecture_room/Lecture room/video (8).avi) |
| 11 | `lecture_room` | **video (9).avi** | 37.4s | **32.36s** | **809** | 30.24s - 32.44s | **46.2%** | staggering (50.8%) | falling (46.2%) | 1.77 | [`Open Video`](file:///F:/fall_detection/data/videos/le2i/Lecture_room/Lecture room/video (9).avi) |
| 12 | `office` | **video (12).avi** | 11.92s | **9.04s** | **226** | 7.48s - 9.44s | **76.7%** | falling (76.7%) | staggering (11.7%) | 1.68 | [`Open Video`](file:///F:/fall_detection/data/videos/le2i/Office/Office/video (12).avi) |
| 13 | `office` | **video (13).avi** | 31.68s | **27.8s** | **695** | 26.12s - 28.28s | **28.8%** | falling (28.8%) | staggering (18.9%) | 1.74 | [`Open Video`](file:///F:/fall_detection/data/videos/le2i/Office/Office/video (13).avi) |
| 14 | `office` | **video (16).avi** | 17.12s | **10.8s** | **270** | 9.40s - 11.04s | **21.7%** | staggering (26.7%) | falling (21.7%) | 1.21 | [`Open Video`](file:///F:/fall_detection/data/videos/le2i/Office/Office/video (16).avi) |
| 15 | `office` | **video (17).avi** | 8.68s | **5.76s** | **144** | 3.96s - 6.44s | **22.5%** | staggering (46.9%) | falling (22.5%) | 2.72 | [`Open Video`](file:///F:/fall_detection/data/videos/le2i/Office/Office/video (17).avi) |
| 16 | `office` | **video (2).avi** | 10.36s | **7.48s** | **187** | 4.88s - 7.80s | **88.4%** | falling (88.4%) | wear a shoe (2.4%) | 1.12 | [`Open Video`](file:///F:/fall_detection/data/videos/le2i/Office/Office/video (2).avi) |
| 17 | `office` | **video (4).avi** | 12.8s | **8.72s** | **218** | 6.92s - 8.92s | **54.8%** | falling (54.8%) | wear a shoe (15.7%) | 1.11 | [`Open Video`](file:///F:/fall_detection/data/videos/le2i/Office/Office/video (4).avi) |
| 18 | `office` | **video (5).avi** | 9.52s | **6.48s** | **162** | 5.88s - 6.96s | **89.2%** | falling (89.2%) | wear a shoe (1.8%) | 1.66 | [`Open Video`](file:///F:/fall_detection/data/videos/le2i/Office/Office/video (5).avi) |
| 19 | `office` | **video (6).avi** | 16.6s | **13.8s** | **345** | 11.56s - 14.00s | **32.1%** | staggering (45.0%) | falling (32.1%) | 1.37 | [`Open Video`](file:///F:/fall_detection/data/videos/le2i/Office/Office/video (6).avi) |
| 20 | `office` | **video (7).avi** | 8.68s | **5.72s** | **143** | 3.12s - 5.80s | **58.4%** | falling (58.4%) | staggering (25.5%) | 2.19 | [`Open Video`](file:///F:/fall_detection/data/videos/le2i/Office/Office/video (7).avi) |
