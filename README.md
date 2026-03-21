# Mario Kart 8 LAN Play Video Processor

This program analyzes recordings of Mario Kart 8 LAN-play tournament sessions and turns the detected race results into Excel output.

In practice it:
- scans videos for race score screens and total score screens
- extracts player names, points, positions, tracks, and characters
- rebuilds tournament progress race by race
- exports the results into structured workbook files for review and sharing

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

Short setup guide for Windows.

Linux or macOS? Read [docs/LINUX_MACOS_SETUP.md](./docs/LINUX_MACOS_SETUP.md).

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
- multi-video initial scan now defaults to `2` workers only when EasyOCR CUDA is available
- when EasyOCR CUDA is unavailable, multi-video initial scan defaults back to `1` worker
- `MK8_PARALLEL_VIDEO_SCAN_WORKERS` can still override this manually
- higher values such as `3` or `4` oversubscribed the machine in local testing and were slower than `2`

Recommended performance profile on the current benchmark laptop:
- `execution_mode=cpu`
- `easyocr_gpu_mode=auto`
- `overlap_ocr_mode=race`
- `overlap_ocr_consumers=2`
- `MK8_PARALLEL_VIDEO_SCAN_WORKERS=2`

This combination is the current best verified throughput profile for the full local tournament benchmark set.

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

When `--subfolders` is used:
- supported videos are discovered recursively under `./Input_Videos/`
- exported frame bundles and Excel/CSV `Video` names include a sanitized relative folder path
- this avoids naming conflicts when different folders contain files with the same base filename

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

## If you want more detail

For the Linux/macOS setup guide, read:
- [LINUX_MACOS_SETUP.md](./docs/LINUX_MACOS_SETUP.md)

## Technical Reference

If you want the pipeline, templates, ROIs, and metadata documented for development or reproduction, read:
- [docs/TECHNICAL_PIPELINE.md](./docs/TECHNICAL_PIPELINE.md)
