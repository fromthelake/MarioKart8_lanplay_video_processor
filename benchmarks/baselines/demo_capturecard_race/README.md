# Demo CaptureCard Race Baseline

This baseline captures the current approved scoped output for the restored demo input video:

- source video: `Input_Videos/Demo_CaptureCard_Race.mp4`
- source command:

```powershell
.\.venv\Scripts\python.exe -m mk8_local_play.main --selection --video Demo_CaptureCard_Race.mp4
```

Validation command:

```powershell
.\.venv\Scripts\python.exe tools\validate_outputs.py --baseline-dir benchmarks\baselines\demo_capturecard_race --prefix Demo_CaptureCard_Race --race-class Demo_CaptureCard_Race
```

Baseline contract:
- workbook rows for `RaceClass=Demo_CaptureCard_Race`
- exported frame files with prefix `Demo_CaptureCard_Race`
- annotated score frames with prefix `annotated_Demo_CaptureCard_Race`

This baseline is intentionally scoped. Do not validate the full `Output_Results/` corpus against this folder without the `--prefix` and `--race-class` filters.
