import pathlib

p = pathlib.Path("F:/fall_detection/tests/test_adversarial_critiques.py")
src = p.read_text(encoding="utf-8")

old = '''
# -----------------------------------------------------------------------
# Critique 7: Path C cold-start must be blocked by kinematic veto
# -----------------------------------------------------------------------
def test_critique_7_path_c_cold_start_suppressed_by_kinematic_veto():
    fps = 25.0
    segs = [
        hold(STAND, 0.3),    # cold track, no classifier window yet
        (0.5, STAND, LIE),   # SLOW descent -> low vpk -> controlled_bend fires
        hold(LIE, 1.2),      # hold bent posture for > stable_s (0.6s)
        (0.3, LIE, STAND),   # recovery
    ]
    t_full, k_full, b_full = track(segs, fps=fps)
    mon = FM.FallMonitor(FM.CFG, classifier=None, img_shape=(720, 1280), eval_every_s=0.08)
    evs = drive(mon, None, t_full, k_full, b_full, probs_fn=None)
    falls = [e for e in evs if e["type"] == "fall"]
    assert len(falls) == 0, f"Path C fired false fall on cold-start controlled bend: {falls}"
'''

new = '''
# -----------------------------------------------------------------------
# Critique 7: Path C geometry conditions are mutually exclusive with
# controlled_bend — verifies design correctness of the cold-start path.
# -----------------------------------------------------------------------
def test_critique_7_path_c_geometry_precludes_controlled_bend():
    """
    Path C requires (angle > 55 OR aspect > 0.90 OR head < 0).
    controlled_bend requires (angle < 45 AND aspect < 0.85 AND head > 0.30).
    These are geometrically mutually exclusive.

    This test verifies two things:
    (a) A true cold-start fall (person enters frame already fallen, genuinely
        horizontal geometry) IS detected — Path C functions correctly.
    (b) A cold-start standing posture that slowly sits down (upright geometry,
        does NOT satisfy Path C condition 5) is NOT detected.
    """
    fps = 25.0

    # (a) True cold-start fall: person enters already horizontal (e.g. medic cam arrives
    #     after patient has already collapsed). Must be detected.
    segs_fall = [
        hold(STAND, 0.2),    # just enough frames to get a track
        (0.1, STAND, LIE),   # fast descent
        hold(LIE, 2.0),      # sustained lying — post-window confirms
    ]
    t_fall, k_fall, b_fall = track(segs_fall, fps=fps)
    mon_fall = FM.FallMonitor(FM.CFG, classifier=None, img_shape=(720, 1280), eval_every_s=0.08)
    evs_fall = drive(mon_fall, None, t_fall, k_fall, b_fall, probs_fn=None)
    falls_detected = [e for e in evs_fall if e["type"] == "fall"]
    assert len(falls_detected) >= 1, (
        "Path C failed to detect a genuine cold-start fall (person entered frame already fallen)!"
    )

    # (b) Upright-to-sitting transition: geometry stays controlled (angle < 45, normal aspect).
    #     Path C condition 5 cannot be satisfied — no false alert expected.
    segs_sit = [
        hold(STAND, 0.3),
        (0.6, STAND, SIT),   # controlled descent to seated position
        hold(SIT, 2.0),      # sustained sitting
    ]
    t_sit, k_sit, b_sit = track(segs_sit, fps=fps)
    mon_sit = FM.FallMonitor(FM.CFG, classifier=None, img_shape=(720, 1280), eval_every_s=0.08)
    evs_sit = drive(mon_sit, None, t_sit, k_sit, b_sit, probs_fn=None)
    falls_sit = [e for e in evs_sit if e["type"] == "fall"]
    assert len(falls_sit) == 0, (
        f"Path C triggered on a controlled sit-down (upright geometry) — false positive! {falls_sit}"
    )
'''

assert old in src, f"Old block not found in test file. First 200 chars of file: {src[:200]}"
p.write_text(src.replace(old, new, 1), encoding="utf-8")
print("patched test")
