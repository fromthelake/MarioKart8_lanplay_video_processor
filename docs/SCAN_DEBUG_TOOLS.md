# Scan Debug Tools

This document describes the scan/debug tools currently in `tools/` for candidate detection, RaceScore/Transition/TotalScore detail tracing, and transition-pattern experiments.

## Primary Tools (Keep)

- `tools/candidate_scan_debug_gui.py`
  - Purpose: debug the **initial candidate scan** phase.
  - Scope: production-checked frames on the initial 3s stride.
  - Shows: ignore/score/track/race checks, thresholds, skip decisions, candidate creation.
  - Use when: candidate frame detection is wrong or inconsistent.

- `tools/score_detail_debug_gui.py`
  - Purpose: debug **per-race detail selection** after candidate detection.
  - Modes: `RaceScore Detail`, `Transition Scan`, `TotalScore Scan`.
  - Shows: clickable summary frames, production-step playback, live ROI overlays, trigger progress, stable-run counters, expected/max player diagnostics.
  - Use when: transition frame, points-anchor frame, or total-anchor frame are wrong.

## Supporting CLI Probes (Keep)

- `tools/probe_score_transition.py`
  - Purpose: quick CSV probe for one race in one video.
  - Outputs:
    - `<video>_race_<NNN>_score_probe.csv`
    - `<video>_race_<NNN>_transition_probe.csv`
  - Use when: you need raw frame-by-frame logs without GUI.

- `tools/experimental/transition_pattern_analyzer.py`
  - Purpose: evaluate alternative transition/resort streak patterns (for example p5/p7/p9 with false-gap tolerance).
  - Outputs:
    - `<video>_pattern_analysis.json`
    - `<video>_pattern_summary.csv`
  - Use when: testing new transition pattern logic before production changes.

- `tools/experimental/position_roi_debug_viewer.py`
  - Purpose: inspect position ROI tiles and row-template matching decisions on single frames.
  - Use when: tuning ROI geometry and position-template thresholds.

## Legacy Tool (Superseded)

- `tools/scan_debug_gui.py`
  - Status: legacy/superseded by `tools/score_detail_debug_gui.py`.
  - Reason: overlapping scope with fewer diagnostics and less complete UI behavior.
  - Kept for now to avoid breaking existing local workflows; new debugging should use `score_detail_debug_gui.py`.

## Typical Commands

Candidate scan GUI:

```powershell
.\.venv\Scripts\python.exe tools\candidate_scan_debug_gui.py
```

Detail scan GUI:

```powershell
.\.venv\Scripts\python.exe tools\score_detail_debug_gui.py
```

Transition CSV probe (single race):

```powershell
.\.venv\Scripts\python.exe tools\probe_score_transition.py --video "Mario Kart Toernooien/Level Level/2023-10-12/Toernooi 1 - Ronde 2 - Divisie 1.mp4" --subfolders --race 8
```

Pattern analyzer:

```powershell
.\.venv\Scripts\python.exe tools\experimental\transition_pattern_analyzer.py --video "Input_Videos\\3Races.mp4" --patterns 5,7,9 --max-false-gap 2
```
