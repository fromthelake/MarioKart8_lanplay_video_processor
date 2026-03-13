# Linux and macOS Setup

Short setup guide for Linux and macOS.

GitHub:
- https://github.com/fromthelake/MarioKart8_lanplay_video_processor

## Important

For this project itself:
- everything runs from the local `.venv` inside this project folder
- do not install this app globally with `pip install ...`
- do not add `mk8-local-play` to your shell PATH
- always run the app from this project folder by using the local `.venv` command:
  - `.venv/bin/mk8-local-play`

System-wide installs are only for external tools such as:
- Git
- Python 3.12
- Tesseract OCR

## Step 1. Choose where the project should live

Open a terminal in the parent folder where you want Git to create the project folder.

Examples:
- `~/Projects`
- `~/Documents`

Important:
- the `git clone` command in Step 5 will create a new folder named `MarioKart8_lanplay_video_processor` inside the folder you opened

## Step 2. Check Git

Run:

Terminal Command:
--------------
git --version
--------------

If it works:
- continue to Step 3

If it fails:
- install Git with your system package manager or developer tools
- then run `git --version` again

Typical install commands:

Linux:
Terminal Command:
--------------
sudo apt-get update
sudo apt-get install git
--------------

macOS:
- run `git --version` and allow the Command Line Tools install if prompted

## Step 3. Check Python 3.12

Run:

Terminal Command:
--------------
python3.12 --version
--------------

If `python3.12 --version` shows Python 3.12:
- continue to Step 4

If Python is missing or not Python 3.12:
- install Python 3.12
- then open a new terminal and run `python3.12 --version` again

Typical install commands:

Linux:
Terminal Command:
--------------
sudo apt-get update
sudo apt-get install python3.12 python3.12-venv python3-pip
--------------

macOS:
Terminal Command:
--------------
brew install python@3.12
--------------

Important:
- this installs Python on your system
- the Mario Kart tool itself is still installed only inside this project folder's local `.venv`
- you do not need a global install of `mk8-local-play`

## Step 4. Check Tesseract OCR

Run:

Terminal Command:
--------------
tesseract --version
--------------

If it works:
- continue to Step 5

If it fails:
- install Tesseract
- then run `tesseract --version` again

Typical install commands:

Linux:
Terminal Command:
--------------
sudo apt-get update
sudo apt-get install tesseract-ocr
--------------

macOS:
Terminal Command:
--------------
brew install tesseract
--------------

Optional:
- install `ffmpeg` only if you want merge-video features or manual repair workflows

Linux:
Terminal Command:
--------------
sudo apt-get update
sudo apt-get install ffmpeg
--------------

macOS:
Terminal Command:
--------------
brew install ffmpeg
--------------

## Step 5. Download the project

Run:

Terminal Command:
--------------
git clone https://github.com/fromthelake/MarioKart8_lanplay_video_processor
cd MarioKart8_lanplay_video_processor
--------------

## Step 6. Run setup

Run:

Terminal Command:
--------------
chmod +x ./scripts/setup_unix.sh
./scripts/setup_unix.sh
--------------

This setup script:
- creates or reuses the local `.venv` in this project folder
- uses `python3.12` by default and stops if the interpreter is not Python 3.12
- installs the app into that local `.venv`
- does not require a global install of this app
- does not require adding `mk8-local-play` to PATH

If setup succeeds:
- continue to Step 7

If setup fails:
- read the terminal error
- if the script reports the wrong Python version, delete `.venv`, set `PYTHON_BIN` to Python 3.12, and rerun it
- fix any other missing dependency
- run `./scripts/setup_unix.sh` again

## Step 7. Run the environment check

Run:

Terminal Command:
--------------
.venv/bin/mk8-local-play --check
--------------

If the check succeeds:
- continue to Step 8

If the check says Tesseract is missing:
- install Tesseract for your platform
- run the check again

If the check succeeds, the project is ready to run entirely from:
- `.venv/bin/mk8-local-play`

## Step 8. Add your videos

Put your video files in folder:

`./Input_Videos/`

Optional:
- you can also place videos inside subfolders under `./Input_Videos/`
- use `--subfolders` if you want headless runs to include those subfolders

## Step 9. Run the tool

Process everything in `Input_Videos`:

Terminal Command:
--------------
.venv/bin/mk8-local-play --all
--------------

Process everything in `Input_Videos` and all subfolders:

Terminal Command:
--------------
.venv/bin/mk8-local-play --all --subfolders
--------------

Process only the current selected input set, including subfolders:

Terminal Command:
--------------
.venv/bin/mk8-local-play --selection --subfolders
--------------

When `--subfolders` is used:
- supported videos are discovered recursively under `./Input_Videos/`
- exported frame bundles and Excel/CSV `Video` names include a sanitized relative folder path
- this avoids naming conflicts when different folders contain files with the same base filename

Process only the current selected input set:

Terminal Command:
--------------
.venv/bin/mk8-local-play --selection
--------------

## Output

Results are written to folder:

`./Output_Results/`

## Commands

Open the GUI interface:

Terminal Command:
--------------
.venv/bin/mk8-local-play
--------------

What it does:
- starts the desktop GUI
- from the GUI you can open folders, run extraction, run OCR, and export results

Run everything:

Terminal Command:
--------------
.venv/bin/mk8-local-play --all
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

Terminal Command:
--------------
.venv/bin/mk8-local-play --all --subfolders
--------------

What it changes:
- extraction also includes supported `.mp4`, `.mkv`, `.mkv`, `.mov`, `.avi`, and `.webm` files found in subfolders under `Input_Videos`
- OCR/export still behaves like `--all`, so existing historical frame groups can still be included

Run only the current selected input set:

Terminal Command:
--------------
.venv/bin/mk8-local-play --selection
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

Terminal Command:
--------------
.venv/bin/mk8-local-play --selection --subfolders
--------------

What it changes:
- extraction includes the current selected input set across `Input_Videos` and its subfolders
- OCR/export stays scoped to only those subfolder-aware video classes

Run extraction only:

Terminal Command:
--------------
.venv/bin/mk8-local-play --extract
--------------

What it does:
- scans videos and exports frame bundles into `Output_Results/Frames`
- does not run OCR or create the final workbook

Run OCR/export only:

Terminal Command:
--------------
.venv/bin/mk8-local-play --ocr
--------------

What it does:
- runs OCR on the extracted frames currently present in `Output_Results/Frames`
- writes the workbook output

What it does not do:
- it does not extract frames from videos first

Run one video only with scoped OCR:

Terminal Command:
--------------
.venv/bin/mk8-local-play --selection --video Demo_CaptureCard_Race.mp4
--------------

What it does:
- extracts only that one video
- limits OCR/export to that same video class

Recommended use:
- use this when you want a true one-video run
- prefer this over `--all --video ...`, because `--all` can still include older frame groups during OCR

## Troubleshooting

First try these checks:

Terminal Command:
--------------
git --version
python3.12 --version
tesseract --version
.venv/bin/mk8-local-play --check
--------------

Then read:
- [README.md](../README.md)
- [TECHNICAL_PIPELINE.md](./TECHNICAL_PIPELINE.md)
