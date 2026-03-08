Mario Kart 8 Local Play Video Processor

This tool analyzes Mario Kart 8 local play vertical split-screen videos, finds the important race result screens, reads the player names and points, and exports the results to an Excel file.

In short, it does this:
1. scans videos in `Input_Videos`
2. saves the important screenshots for each race
3. OCRs those screenshots
4. writes `Output_Results/Tournament_Results.xlsx`

It supports:
- Windows GUI usage
- Windows/Linux/macOS command-line usage
- optional runtime configuration without editing Python files

Quick Start

If you just want to try it:

Windows:

```powershell
pip install -r requirements.txt
python Main_RunMe.py --check
python Main_RunMe.py --all
```

Linux:

```bash
pip install -r requirements.txt
python3 Main_RunMe.py --check
python3 Main_RunMe.py --all
```

When it finishes, open:
- `Output_Results/Tournament_Results.xlsx`

Fastest First-Time Setup

If you want the easiest setup path, use one of the included setup scripts.

Windows PowerShell:

```powershell
.\scripts\setup_windows.ps1
```

Linux/macOS:

```bash
./scripts/setup_unix.sh
```

These scripts:
- create a virtual environment if needed
- install Python dependencies
- create `app_config.json` from the example file if missing
- run `Main_RunMe.py --check`

What Kind Of Video Is Expected

The project is built for:
- Mario Kart 8 local play
- vertical split-screen layout
- result screens that look like the included sample material

Best results usually come from:
- clear capture-card footage
- stable resolution
- visible full game image
- normal scoreboard/result screens without overlays

Less reliable inputs:
- heavily compressed footage
- unusual crops
- missing borders / partial game capture
- streams with large overlays, alerts, or webcam boxes

What The Tool Creates

Main result:
- `Output_Results/Tournament_Results.xlsx`

Intermediate screenshots:
- `Output_Results/Frames`

Optional debug output:
- `Output_Results/Debug/debug_max_val.csv`
- `Output_Results/Debug/Score_Frames`
- `Output_Results/Debug/linking_*.xlsx`

The screenshots in `Output_Results/Frames` are the key race screens the tool selected:
- `0TrackName`
- `1RaceNumber`
- `2RaceScore`
- `3TotalScore`

Requirements

1. Python
- Python 3.10+ recommended

Install Python packages:

```powershell
pip install -r requirements.txt
```

2. Tesseract OCR
- Required for reading player names and text

Windows:
- install Tesseract OCR, for example the UB Mannheim build

Linux:

```bash
sudo apt-get install tesseract-ocr
```

macOS:

```bash
brew install tesseract
```

3. FFmpeg
- Only required for the optional video merge feature

Windows:
- install FFmpeg and make sure `ffmpeg` works from a terminal

Linux:

```bash
sudo apt-get install ffmpeg
```

macOS:

```bash
brew install ffmpeg
```

Check Your Setup

Before running the pipeline, use:

```powershell
python Main_RunMe.py --check
```

or:

```bash
python3 Main_RunMe.py --check
```

This prints:
- whether Tesseract was found
- whether FFmpeg was found
- where the input and output folders are
- worker counts
- debug settings

Typical Usage

GUI mode on Windows:

```powershell
python Main_RunMe.py
```

Headless mode:

```powershell
python Main_RunMe.py --extract
python Main_RunMe.py --ocr
python Main_RunMe.py --all
```

Linux/macOS:

```bash
python3 Main_RunMe.py --extract
python3 Main_RunMe.py --ocr
python3 Main_RunMe.py --all
```

Recommended Workflow

1. Put your `.mp4` or `.mkv` files into `Input_Videos`
2. Run `--check`
3. Run `--extract` or `--all`
4. Inspect `Output_Results/Frames` if you want to verify the selected race screenshots
5. Run `--ocr` if you did not use `--all`
6. Open `Output_Results/Tournament_Results.xlsx`

Examples

Example 1: full run on Windows

```powershell
python Main_RunMe.py --check
python Main_RunMe.py --all
```

Example 2: split the run into two steps

```powershell
python Main_RunMe.py --extract
python Main_RunMe.py --ocr
```

Example 3: full run on Linux

```bash
python3 Main_RunMe.py --check
python3 Main_RunMe.py --all
```

Configuration

You can configure the runtime in either of these ways:
- environment variables
- an optional `app_config.json` file in the project root

An example file is included:
- `app_config.example.json`

Supported config keys:
- `tesseract_cmd`
- `ocr_workers`
- `score_analysis_workers`
- `write_debug_csv`
- `write_debug_score_images`
- `write_debug_linking_excel`

Example `app_config.json`:

```json
{
  "tesseract_cmd": "/usr/bin/tesseract",
  "ocr_workers": 16,
  "score_analysis_workers": 4,
  "write_debug_csv": true,
  "write_debug_score_images": true,
  "write_debug_linking_excel": true
}
```

Supported environment variables:
- `MK8_TESSERACT_CMD`
- `MK8_OCR_WORKERS`
- `MK8_SCORE_ANALYSIS_WORKERS`
- `MK8_WRITE_DEBUG_CSV`
- `MK8_WRITE_DEBUG_SCORE_IMAGES`
- `MK8_WRITE_DEBUG_LINKING_EXCEL`

If Tesseract is installed but not on `PATH`, the easiest fix is usually:
- set `tesseract_cmd` in `app_config.json`
- or set `MK8_TESSERACT_CMD`

Example Windows config:

```json
{
  "tesseract_cmd": "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
}
```

Benchmark And Validation Scripts

These scripts are mainly for development and regression testing.

PowerShell:

```powershell
.\scripts\quick_benchmark.ps1
.\scripts\release_benchmark.ps1
```

Bash:

```bash
./scripts/quick_benchmark.sh
./scripts/release_benchmark.sh
```

What they do:
- isolate the selected benchmark video
- clear generated outputs
- run `--check`
- run extraction and OCR/export
- validate the result against a stored baseline
- print extraction and OCR timings

Default benchmark videos:
- quick benchmark: `Test_3_Races.mkv`
- release benchmark: `Divisie_1.mkv`

You can also validate manually:

```powershell
python tools\validate_outputs.py --baseline-dir baselines/quick/Test_3_Races
```

GUI Notes

The GUI is mainly for Windows users who want a simple click-through workflow.

The GUI can:
- open the input folder
- run extraction
- open the extracted frames folder
- clear found race screenshots
- run OCR/export
- open the result workbook
- merge multiple clips into one file with FFmpeg

On systems without Tkinter, use the CLI instead.

Git And Generated Files

Generated outputs are intentionally not tracked in git.

That includes:
- extracted race screenshots
- OCR debug images
- `Tournament_Results.xlsx`
- local benchmark outputs
- local large input videos

This keeps the repository clean and makes commits focus on code and documentation only.

Troubleshooting

Problem: `Tesseract was not found`
- install Tesseract
- run `python Main_RunMe.py --check`
- if needed, set `tesseract_cmd` in `app_config.json`

Problem: `FFmpeg was not found`
- install FFmpeg
- make sure `ffmpeg` is available on `PATH`
- FFmpeg is only needed for the merge-video feature

Problem: no races were found
- verify the video is Mario Kart 8 local play vertical split-screen
- check whether the full game image is visible
- inspect `Output_Results/Frames`
- try one of the included example-style clips first

Problem: OCR output looks wrong
- inspect `Output_Results/Frames`
- inspect `Output_Results/Debug/Score_Frames`
- poor quality input or unusual overlays can reduce OCR quality

Problem: the GUI does not start on Linux
- use the CLI mode instead:

```bash
python3 Main_RunMe.py --all
```

Problem: normal runs generate too many debug files
- disable debug artifacts in `app_config.json`:

```json
{
  "write_debug_csv": false,
  "write_debug_score_images": false,
  "write_debug_linking_excel": false
}
```

Project Summary

This project is a practical local-play results extractor:
- it finds result screens from long videos
- it OCRs names and track information
- it reconstructs race points
- it exports a tournament-style Excel overview

If you are sharing it with friends, the simplest path is:
1. install Python dependencies
2. install Tesseract
3. run `python Main_RunMe.py --check`
4. put videos in `Input_Videos`
5. run `python Main_RunMe.py --all`
