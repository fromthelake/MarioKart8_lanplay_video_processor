# Mario Kart 8 LAN Play Video Processor

This program analyzes recordings of Mario Kart 8 LAN-play tournament sessions and turns the detected race results into Excel output.

In practice it:
- scans videos for race score screens and total score screens
- extracts player names, points, positions, tracks, and characters
- rebuilds tournament progress race by race
- exports the results into structured workbook files for review and sharing

Current output set for a normal run:
- `*_Tournament_Results.xlsx`
- `*_Tournament_Results.csv`
- `*_Final_Standings.csv`

Debug outputs can be enabled for a scoped headless run with:
- `.\.venv\Scripts\mk8-local-play.exe --selection --debug --video <video-name>`
- `.\.venv\Scripts\mk8-local-play.exe --selection --subfolders --videos "2026-03-28/VideoA.mp4" "2026-03-28/VideoB.mp4"`
- `.\.venv\Scripts\mk8-local-play.exe --ocr --selection --subfolders --videos "Mario Kart Toernooien/Level Level/2023-10-12/Toernooi 1 - Ronde 2 - Divisie 1.mp4" --low_res --debug`

When `--debug` is enabled, the run also writes:
- `Debug/*_Tournament_Results_Debug.xlsx`
- `Debug/*_Tournament_Results_Debug.csv`

Recent scoring and validation behavior:
- explicit multi-video CLI selection is now available through `--videos`, so you can process several exact file paths together in one scoped run
- when `--subfolders` is combined with explicit relative paths in `--videos`, each requested path now resolves exactly instead of also pulling same-named files from other folders such as `backup/`
- score recomputation now resets running tournament totals per video / race class, so repeated player names across separate captures no longer inherit totals from earlier videos
- videos can now contain multiple connection resets; later resets in the same source video are detected and segmented correctly
- reset detection now has a second pass for obvious fresh-session total-score patterns where the displayed totals collapse back to race-points-scale values across most of the field
- temporary player-drop races can stay visible in the workbook while being excluded from tournament totals when a later race recovers to a higher player count
- user exports now include `Counts Toward Totals` and `Scoring Note` at the end of the table when that late scoring policy applies
- first-race scoring recompute now preserves a valid non-zero `OldTotalScore` baseline for the players actually present instead of resetting those totals back to zero
- overlap OCR finalization now ignores incomplete race folders that never exported a `2RaceScore` bundle, so partially scanned tail races no longer block a whole video's workbook rows from appearing in full multi-video runs
- identity standardization now preserves visibly distinct case-only names when they coexist in the same race, so players such as `Floris` and `floris` are not merged into one identity chain
- connection-reset relinking now has a single-swap fallback, so if exactly one player identity changes at reset time it can still relink by elimination even when OCR names are noisy
- one-race low-confidence OCR outlier names are now relinked to the stable adjacent-race identity when continuity proves they are the same player
- headless runs now support experimental `--low_res` mode for explicitly selected videos, forcing those race classes through the existing low-res/ultra-low-res identity path without changing default behavior (`--ultra_low_res` remains as a backward-compatible alias)
- recursive runs now skip any videos under a folder named `corrupt` or `exclude`
- final-race duplicate-name ambiguity notes now only mark the rows that are still truly interchangeable, and the note names the conflicting identity label(s)
- score detection now uses the left-side row-box position signal for the required visible-player prefix instead of relying on a standalone score-strip template match
- initial score confirmation now treats rows `2..6` as the required visible-player prefix, so Nintendo `Capture taken.` overlays on row `1` no longer suppress real score candidates
- 12th-place checks now support both the legacy and Dutch templates during score selection
- TotalScore timing now waits for a continuous score-signal drop of `5.0 * fps` and anchors from the start of that drop, so short transition animations no longer trigger early TotalScore exports
- points-transition debounce now uses a fixed confirm-hit count (`p5` by default) with an FPS-scaled false-gap tolerance, so high-FPS sources keep equivalent gap tolerance without over-delaying transition confirmation
- second-pass score selection now uses a coarse search with rewind before the first hit and again during TotalScore stabilization, reducing wasted frame-by-frame scans
- RaceScore export bundles are now centered on the detected score-transition frame, and the saved `2RaceScore` frames are reused directly by OCR
- the OCR position-template matcher now uses the masked `Score_template_white.png` / `Score_template_black.png` tile path only

Current score-screen support:
- LAN 2 two-player split-screen score layouts
- LAN 1 one-player full-screen score layouts

The score-screen pipeline now auto-detects the supported score layout during extraction.
For `2RaceScore` and `3TotalScore`, exported frame names and metadata carry the detected
layout tag so OCR can use the matching ROI set directly.

Character OCR also now includes a conservative session-level Mii fallback:
- when one stable player identity repeatedly produces weak, near-tied non-Mii character matches
- and those winning non-Mii matches are unstable across races
- the exported character is relabeled to `Mii`
- the row receives a short review note: `mii_fallback_unstable_character_match`

Character OCR now also includes a roster-family variant refinement pass before that fallback:
- catalog-backed color-variant families such as `Birdo`, `Yoshi`, `Shy Guy`, and `Inkling` are rescored only against members of the same family
- explicit close-cutout families such as `Peach` / `Pink Gold Peach` and `Mario` / `Metal Mario` / `Gold Mario` are also compared inside their own family groups
- the default/base roster member stays in the family comparison instead of being treated separately
- the refinement uses the same aligned alpha-cutout color scoring as character matching, across the calibrated local alignment offsets, from the saved RaceScore anchor frame
- this is intended to stabilize true family members before the conservative `Mii` fallback is allowed to relabel them

Family-variant debug probe on saved character crops:
- `.\.venv\Scripts\python.exe tools\evaluate_character_variant_families.py --crop-dir Output_Results\Debug\character_probe_20260328`

Short setup guide for Windows.

Linux or macOS? Read [docs/LINUX_MACOS_SETUP.md](./docs/LINUX_MACOS_SETUP.md).

Scan/debug tooling reference: [docs/SCAN_DEBUG_TOOLS.md](./docs/SCAN_DEBUG_TOOLS.md).

GitHub:
- https://github.com/fromthelake/MarioKart8_lanplay_video_processor

## Important

For this project itself:
- everything runs from the local `.venv` inside this project folder
- do not install this app globally with `pip install ...`
- do not add `mk8-local-play` to your system PATH
- always run the app from this project folder by using the local `.venv` command:
  - `.\.venv\Scripts\mk8-local-play.exe` on Windows
  - `.venv/bin/mk8-local-play` on Linux/macOS

System-wide installs are only for external tools such as:
- Git
- Python 3.12
- FFmpeg

## Step 1. Choose where the project should live

Choose the folder where you want GitHub to create the project folder.

Example:
- Desktop
- Documents
- a development folder such as `C:\Projects`

Open that parent folder in File Explorer.

Then open PowerShell there:
- hold `Shift`
- right-click in the folder background
- click `Open PowerShell window here` or `Open in Terminal`

Important:
- the `git clone` command in Step 4 will create a new folder named `MarioKart8_lanplay_video_processor` inside the folder you opened

## Step 2. Check Git

Run:

PowerShell Command:
--------------
git --version
--------------

If it works:
- continue to Step 3

If it fails:
- download and install Git for Windows:
  - https://git-scm.com/download/win
- open a new PowerShell window
- run `git --version` again

## Step 3. Check Python 3.12

Run:

PowerShell Command:
--------------
python --version
--------------

If that does not show Python 3.12, run:

PowerShell Command:
--------------
py -3.12 --version
--------------

If either command shows Python 3.12:
- continue to Step 4

If Python 3.12 is missing:
- on most Windows 10/11 systems, first try:

PowerShell Command:
--------------
winget install Python.Python.3.12
--------------

- if `winget` is not available or fails, download Python 3.12 manually from:
  - https://www.python.org/downloads/windows/
- use Python 3.12 exactly for setup; newer Python versions such as 3.13 or 3.14 are not supported yet
- during install, enable `Add Python to PATH` if shown
- open a new PowerShell window
- run `py -3.12 --version` again

Important:
- this installs Python on your system
- the Mario Kart tool itself is still installed only inside this project folder's local `.venv`
- you do not need a global install of `mk8-local-play`

## Step 4. Download the project

Run:

PowerShell Command:
--------------
git clone https://github.com/fromthelake/MarioKart8_lanplay_video_processor
cd MarioKart8_lanplay_video_processor
--------------

## Step 5. Run setup

Run:

PowerShell Command:
--------------
.\scripts\setup_windows.ps1
--------------

This setup script:
- creates or reuses the local `.venv` in this project folder
- uses Python 3.12 specifically and stops if only a newer Python is installed
- installs the app into that local `.venv`
- installs the Python OCR dependencies, including EasyOCR
- does not require a global install of this app
- does not require adding `mk8-local-play` to PATH

If setup succeeds:
- continue to Step 6

If setup fails:
- read the error shown in PowerShell
- fix the missing dependency
- run `./scripts/setup_windows.ps1` again

## Step 6. Run the environment check

Run:

PowerShell Command:
--------------
.\.venv\Scripts\mk8-local-play.exe --check
--------------

If the check succeeds:
- continue to Step 7

If the check succeeds, the project is ready to run entirely from:
- `.\.venv\Scripts\mk8-local-play.exe`

Screenshot export format:
- extracted screenshots are controlled by `config/app_config.json` -> `export_image_format`
- accepted values are `jpg`, `jpeg`, and `png`
- the current default is `jpg` for smaller exported frame files
- use `png` if you want lossless frame exports for troubleshooting or comparison work
- `MK8_EXPORT_IMAGE_FORMAT` can still override the config for a single run

Headless debug toggle:
- normal CLI runs can stay lean and skip debug workbook/image output
- use `--debug` on `mk8-local-play.exe` or `python -m mk8_local_play.main` when you explicitly want debug CSV, debug workbook, and score-layout images for investigation

Runtime GPU mode defaults:
- `config/app_config.json` now defaults `execution_mode` to `cpu` and `easyocr_gpu_mode` to `auto`
- `execution_mode` controls OpenCV extraction acceleration and accepts `auto`, `gpu`, or `cpu`
- `easyocr_gpu_mode` controls EasyOCR and accepts `auto`, `gpu`, or `cpu`
- extraction defaults to `cpu` because that is the fastest verified setting on this machine profile
- in `auto`, extraction uses CUDA when available and otherwise falls back to CPU
- OpenCL extraction remains available through explicit `GPU` mode, but is not chosen automatically
- when EasyOCR is using GPU, effective OCR workers stay at `1`
- `overlap_ocr_mode` now defaults to `auto`
- `overlap_ocr_consumers` now defaults to `2`
- in overlap `auto`, multi-video full runs use the streamed per-race overlap path only when EasyOCR CUDA is available; otherwise runs stay on the existing sequential path
- you can still override overlap mode to `video` or `race`, and raise `overlap_ocr_consumers` later for experiments
- multi-video initial scan now defaults to `2` workers for multi-video runs
- `MK8_PARALLEL_VIDEO_SCAN_WORKERS` can still override this manually
- higher values such as `3` or `4` oversubscribed the machine in local testing and were slower than `2`

Recommended performance profile on the current benchmark laptop:
- `execution_mode=cpu`
- `easyocr_gpu_mode=auto`
- `overlap_ocr_mode=race`
- `overlap_ocr_consumers=2`
- `MK8_PARALLEL_VIDEO_SCAN_WORKERS=2`

Additional extraction defaults now tuned from the full 7-video benchmark:
- `pass1_scan_workers=4`
- `score_analysis_workers=4`
- `parallel_video_total_score_workers` resolves to `2` on `16+` logical CPU threads and `1` otherwise

This combination is the current best verified throughput profile for the full local tournament benchmark set.

Console output during a run now uses a clearer live format:
- each video gets a stable neon accent color for the whole run
- labels stay neutral while video-owned values are colorized
- workflow ordering is consistent across the input summary, frame-count preflight, scan, and per-video summaries
- scan progress now shows `HH:MM:SS / HH:MM:SS` instead of raw frame counters
- live progress uses aligned `Comp` / `Done` fields and includes CPU/RAM/GPU where useful for stall detection
- RAM in live progress and phase summaries is reported as percentage
- confirmed scan detections list `Race`, `Track`, and `Score` anchors in frame order with source time and frame number
- OCR progress uses `Active` for in-flight race bundles and overlap queue labels use `Que` / `AllQue`
- the final performance summary uses aligned tables for run totals, split phase timings, per-video status, resource peaks, and video-seconds-per-wall-second rate
- `Time saved by overlap` shows the wall-clock time saved through overlap and parallelism

Placeholder identity handling is now tiered:
- normal placeholder rescue still requires repeated multi-race support
- if that fails, a conservative forced-choice fallback can promote a strong top candidate
- forced promotions are marked in the review trail with `placeholder_name_forced_choice`

## Step 7. Add your videos

Put your video files in folder:

`./Input_Videos/`

Optional:
- you can also place videos inside subfolders under `./Input_Videos/`
- use `--subfolders` if you want headless runs to include those subfolders

## Step 8. Run the tool

Process everything in `Input_Videos`:

PowerShell Command:
--------------
.\.venv\Scripts\mk8-local-play.exe --all
--------------

Process everything in `Input_Videos` and all subfolders:

PowerShell Command:
--------------
.\.venv\Scripts\mk8-local-play.exe --all --subfolders
--------------

Process only the current selected input set, including subfolders:

PowerShell Command:
--------------
.\.venv\Scripts\mk8-local-play.exe --selection --subfolders
--------------

Process a specific multi-video set by explicit relative file path:

PowerShell Command:
-------------
.\.venv\Scripts\mk8-local-play.exe --selection --subfolders --videos "2026-03-28/Kwalificatie_Groep_1_2026-03-27 20-00-33.mkv" "2026-03-28/Kwalificatie_Groep_2_2026-03-27 20-00-33.mp4" "2026-03-28/Kwalificatie_Groep_3_2026-03-27 20-00-33.mkv"
-------------

When `--subfolders` is used:
- supported videos are discovered recursively under `./Input_Videos/`
- exported frame bundles and Excel/CSV `Video` names include a sanitized relative folder path
- this avoids naming conflicts when different folders contain files with the same base filename
- with `--videos`, explicit relative paths are matched exactly before filename fallback is attempted

Process only the current selected input set:

PowerShell Command:
--------------
.\.venv\Scripts\mk8-local-play.exe --selection
--------------

## Output

Results are written to folder:

`./Output_Results/`

Extracted race screenshots are written under:

`./Output_Results/Frames/`

Their file extension follows `config/app_config.json` -> `export_image_format`.
Examples:
- `Output_Results/Frames/Demo_CaptureCard_Race/Race_001/0TrackName.jpg`
- `Output_Results/Frames/Demo_CaptureCard_Race/Race_001/1RaceNumber.jpg`
- `Output_Results/Frames/Demo_CaptureCard_Race/Race_001/2RaceScore/anchor_5869.jpg`
- `Output_Results/Frames/Demo_CaptureCard_Race/Race_001/2RaceScore/consensus_5866.jpg`
- `Output_Results/Frames/Demo_CaptureCard_Race/Race_001/3TotalScore/anchor_5994.jpg`

Important:
- score-screen OCR now persists the full frame bundles it uses
- both `--selection` and `--ocr` read the same saved score bundles
- `anchor_<frame>.jpg` is the exported anchor frame
- `consensus_<frame>.jpg` files are the neighboring OCR-vote frames used for that score screen

## Commands

Open the GUI interface:

PowerShell Command:
--------------
.\.venv\Scripts\mk8-local-play.exe
--------------

What it does:
- starts the desktop GUI
- from the GUI you can:
  - open the input folder
  - merge videos
  - run extraction only
  - run a scoped selection pass
  - toggle subfolder-aware processing
  - run OCR/export only
  - open the latest Excel output
  - clear extracted races or output results

GUI command mapping:
- `Find Races In Videos`
  - finds and saves the race screens from your videos
- `Run Selected Videos`
  - does both steps in one go, but only for the selected videos
- `Also Look In Subfolders`
  - includes videos stored in folders inside `Input_Videos`
- `Create Excel Results`
  - reads the saved race screens and creates the Excel file

Run everything:

PowerShell Command:
--------------
.\.venv\Scripts\mk8-local-play.exe --all
--------------

What it does:
- runs extraction on all videos currently present in `Input_Videos`
- then runs OCR/export on all frames present in `Output_Results/Frames`

What it includes:
- the current videos in `Input_Videos`
- existing extracted frames already present in `Output_Results/Frames`

What it does not do:
- it does not limit OCR to only newly extracted frames

Add subfolders to `--all`:

PowerShell Command:
--------------
.\.venv\Scripts\mk8-local-play.exe --all --subfolders
--------------

What it changes:
- extraction also includes supported `.mp4`, `.mkv`, `.mkv`, `.mov`, `.avi`, and `.webm` files found in subfolders under `Input_Videos`
- OCR/export still behaves like `--all`, so existing historical frame groups can still be included

Run only the current selected input set:

PowerShell Command:
--------------
.\.venv\Scripts\mk8-local-play.exe --selection
--------------

What it does:
- runs extraction on the currently selected videos in `Input_Videos`
- then runs OCR/export only for those same video classes

What it includes:
- only the selected/current input videos for this run
- only OCR groups that belong to those selected videos

What it does not do:
- it does not sweep unrelated historical frame groups from older videos

Add subfolders to `--selection`:

PowerShell Command:
--------------
.\.venv\Scripts\mk8-local-play.exe --selection --subfolders
--------------

What it changes:
- extraction includes the current selected input set across `Input_Videos` and its subfolders
- OCR/export stays scoped to only those subfolder-aware video classes

Run extraction only:

PowerShell Command:
--------------
.\.venv\Scripts\mk8-local-play.exe --extract
--------------

What it does:
- scans videos and exports frame bundles into `Output_Results/Frames`
- does not run OCR or create the final workbook

Run OCR/export only:

PowerShell Command:
--------------
.\.venv\Scripts\mk8-local-play.exe --ocr
--------------

What it does:
- runs OCR on the extracted frames currently present in `Output_Results/Frames`
- writes the workbook output

What it does not do:
- it does not extract frames from videos first

Run OCR/export only, but scoped like `--selection`:

PowerShell Command:
--------------
.\.venv\Scripts\mk8-local-play.exe --selection --ocr
--------------

What it does:
- runs OCR only for the video classes currently selected in `Input_Videos`
- ignores unrelated historical frame groups from other videos

Run one video only with scoped OCR:

PowerShell Command:
--------------
.\.venv\Scripts\mk8-local-play.exe --selection --video Demo_CaptureCard_Race.mp4
--------------

What it does:
- extracts only that one video
- limits OCR/export to that same video class

Recommended use:
- use this when you want a true one-video run
- prefer this over `--all --video ...`, because `--all` can still include older frame groups during OCR

Run several exact videos together with scoped OCR:

PowerShell Command:
-------------
.\.venv\Scripts\mk8-local-play.exe --selection --subfolders --videos "2026-03-28/Kampioen_2026-03-27 21-50-56.mp4" "2026-03-28/Talent_2026-03-27 21-50-56.mp4" "2026-03-28/Wild_2026-03-27 21-50-56.mp4"
-------------

What it does:
- extracts only those explicitly listed files
- limits OCR/export to those same video classes
- keeps multi-video overlap OCR available, so CUDA-backed EasyOCR can still process the selected set together

## If you want more detail

For the Linux/macOS setup guide, read:
- [LINUX_MACOS_SETUP.md](./docs/LINUX_MACOS_SETUP.md)

## Technical Reference

If you want the pipeline, templates, ROIs, and metadata documented for development or reproduction, read:
- [docs/TECHNICAL_PIPELINE.md](./docs/TECHNICAL_PIPELINE.md)
