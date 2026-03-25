# Release Checklist

Use this checklist before calling the repo release-ready.

## Code

- Run `python -m compileall mk8_local_play`
- Run `python -m mk8_local_play.main --check`
- Run `python -m mk8_local_play.main --extract --video Demo_CaptureCard_Race.mp4`
- Run `python -m mk8_local_play.main --ocr --video Demo_CaptureCard_Race.mp4`

## Output

- Confirm a timestamped `Output_Results/*_Tournament_Results.xlsx` is created
- Confirm extracted screenshots in `Output_Results/Frames/` look correct
- If output logic changed, compare against a curated baseline
- If running with subfolders enabled, confirm archived `corrupt/` videos and any `exclude/` subtree are not discovered

## Cross-Platform

- Windows setup script still works
- Unix setup script still works
- Headless CLI still works without GUI dependencies
- GUI still starts where Tk is available

## Documentation

- `README.md` is up to date
- `docs/PROJECT_STRUCTURE.md` is up to date
- `docs/CONTRIBUTING.md` is up to date
- `docs/CHANGELOG.md` includes the latest user-visible changes

## Repo Hygiene

- No generated runtime output is accidentally staged
- Placeholder folders still exist for a clean clone
- Config examples still match the real runtime behavior
