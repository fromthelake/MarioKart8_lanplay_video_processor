# Beginner Setup Guide

This guide is for people who have never done this before.

It explains, step by step, how to get the project working on:
- Windows
- Linux
- macOS

The goal is simple:
- install the few required tools
- download the project from GitHub
- run the setup script
- process your videos

If you only want the shortest version, read [README.md](./README.md).

## Before You Start

You need:
- internet access
- a keyboard and mouse
- permission to install software on your computer

You will install or check:
- Git
- Python
- Tesseract OCR

Optional:
- FFmpeg

## What Each Tool Does

- Git
  - downloads the project from GitHub
- Python
  - runs the program
- Tesseract OCR
  - reads names and scores from the screenshots
- FFmpeg
  - only needed if you want to merge multiple video clips into one file

## Windows Guide

### Step 1. Open PowerShell

Easy ways:

1. Press `Windows key + X`
2. Click `Terminal` or `Windows PowerShell`

Alternative:

1. Press `Windows key + R`
2. Type `powershell`
3. Press `Enter`

You should now see a window with text and a blinking cursor.

### Step 2. Check whether Git is installed

Copy and paste this into PowerShell:

```powershell
git --version
```

Expected output:
- something like `git version 2.49.0.windows.1`

If it prints a version number:
- Git is installed
- continue to Step 3

If you get a message like “git is not recognized”:
- Git is not installed yet

Install Git:

1. Open your browser
2. Search for `Git for Windows`
3. Download and install it
4. Close PowerShell
5. Open PowerShell again
6. Run:

```powershell
git --version
```

Expected output:
- something like `git version 2.49.0.windows.1`

### Step 3. Check whether Python is installed

Copy and paste this into PowerShell:

```powershell
python --version
```

Expected output:
- something like `Python 3.12.8`

If that does not work, also try:

```powershell
py --version
```

Expected output:
- something like `Python 3.12.8`

If one of them prints a Python version:
- Python is installed
- continue to Step 4

If neither works:

1. Open your browser
2. Search for `Python 3.12 Windows`
3. Install Python
4. During install, enable the option to add Python to PATH if offered
5. Close PowerShell
6. Open PowerShell again
7. Run:

```powershell
python --version
```

Expected output:
- something like `Python 3.12.8`

### Step 4. Check whether Tesseract is installed

Copy and paste this into PowerShell:

```powershell
tesseract --version
```

Expected output:
- first line looks something like `tesseract 5.5.0`

If it prints a version:
- Tesseract is installed
- continue to Step 5

If not:

1. Open your browser
2. Search for `Tesseract OCR Windows`
3. Install Tesseract
4. Close PowerShell
5. Open PowerShell again
6. Run:

```powershell
tesseract --version
```

Expected output:
- first line looks something like `tesseract 5.5.0`

### Step 5. Choose where you want the project folder

A simple choice is your Desktop.

Move there:

```powershell
cd $HOME\Desktop
```

Expected result:
- no error message
- your PowerShell prompt now points to your Desktop folder

### Step 6. Download the project from GitHub

Replace `<GITHUB-URL>` with the real repository URL:

```powershell
git clone <GITHUB-URL>
```

Expected output:
- lines that mention `Cloning into ...`
- then download progress

Then open the project folder:

```powershell
cd MarioKart8_lanplay_video_processor
```

Expected result:
- no error message
- your PowerShell prompt now points inside the project folder

If the folder has a different name, use that folder name instead.

### Step 7. Run the setup script

Copy and paste this into PowerShell:

```powershell
.\scripts\setup_windows.ps1
```

What this does:
- creates a local Python environment in `.venv`
- installs the project and dependencies
- uses `config/app_config.json` from the repo
- checks the runtime with the real installed command

Expected output:
- lots of installation text
- then a runtime summary
- ideally including `Tesseract: OK`

### Step 8. If PowerShell blocks the script

If Windows says script execution is blocked, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then answer:
- `Y`

Then run the setup script again:

```powershell
.\scripts\setup_windows.ps1
```

### Step 9. Verify the install

After setup, run:

```powershell
.\.venv\Scripts\mk8-local-play.exe --check
```

You want to see:
- `Tesseract: OK`
- `Child script Python: ...\.venv\Scripts\python.exe`

If FFmpeg is missing, that is fine unless you want the merge feature.

### Step 10. Add your videos

Put your Mario Kart videos in:

- `Input_Videos`

You can open it from PowerShell with:

```powershell
explorer .\Input_Videos
```

### Step 11. Run the full tool

Copy and paste this into PowerShell:

```powershell
.\.venv\Scripts\mk8-local-play.exe --all
```

Expected output:
- progress messages for extraction
- progress messages for OCR
- at the end, a path to a timestamped `*_Tournament_Results.xlsx` file

### Step 12. Open the result

The main result file is the newest timestamped workbook in:

- `Output_Results`

Open the output folder with:

```powershell
explorer .\Output_Results
```

Expected result:
- File Explorer opens the output folder

## Linux Guide

These steps assume a typical Debian or Ubuntu style system. Other Linux distributions are similar, but package commands may differ.

### Step 1. Open Terminal

Common ways:

- press `Ctrl + Alt + T`
- or open the applications menu and search for `Terminal`

### Step 2. Check whether Git is installed

Copy and paste this into Terminal:

```bash
git --version
```

Expected output:
- something like `git version 2.43.0`

If it prints a version:
- continue

If not, install it:

```bash
sudo apt-get update
sudo apt-get install git
```

After that, run this again:

```bash
git --version
```

Expected output:
- something like `git version 2.43.0`

### Step 3. Check whether Python is installed

Copy and paste this into Terminal:

```bash
python3 --version
```

Expected output:
- something like `Python 3.12.8`

If it prints a version:
- continue

If not, install Python and venv support:

```bash
sudo apt-get update
sudo apt-get install python3 python3-venv python3-pip
```

After that, run this again:

```bash
python3 --version
```

Expected output:
- something like `Python 3.12.8`

### Step 4. Check whether Tesseract is installed

Copy and paste this into Terminal:

```bash
tesseract --version
```

Expected output:
- first line looks something like `tesseract 5.3.0`

If it prints a version:
- continue

If not, install it:

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

### Step 5. Optional: install FFmpeg

Only needed if you want the merge-video feature:

```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

### Step 6. Choose where you want the project folder

A simple choice:

```bash
cd ~
mkdir -p Projects
cd Projects
```

### Step 7. Download the project

Replace `<GITHUB-URL>` with the real repository URL:

```bash
git clone <GITHUB-URL>
cd MarioKart8_lanplay_video_processor
```

Expected output:
- lines that mention `Cloning into ...`
- then no error when changing into the folder

### Step 8. Run the setup script

Copy and paste this into Terminal:

```bash
chmod +x ./scripts/setup_unix.sh
./scripts/setup_unix.sh
```

Expected output:
- package install text
- then a runtime summary
- ideally including `Tesseract: OK`

### Step 9. Verify the install

Copy and paste this into Terminal:

```bash
.venv/bin/mk8-local-play --check
```

Expected output:
- `Tesseract: OK`
- `Child script Python: .../.venv/bin/python`

### Step 10. Add your videos

Put your files in:

- `Input_Videos/`

You can open the folder in a file manager on many desktops with:

```bash
xdg-open Input_Videos
```

Expected result:
- your file manager opens the input folder

### Step 11. Run the full tool

```bash
.venv/bin/mk8-local-play --all
```

Expected output:
- extraction progress
- OCR progress
- final workbook path

### Step 12. Open the output folder

```bash
xdg-open Output_Results
```

Expected result:
- your file manager opens the output folder

## macOS Guide

### Step 1. Open Terminal

Easy ways:

1. Press `Command + Space`
2. Type `Terminal`
3. Press `Enter`

### Step 2. Check whether Git is installed

Copy and paste this into Terminal:

```bash
git --version
```

Expected output:
- something like `git version 2.39.3`

If it prints a version:
- continue

If it asks to install developer tools:
- allow that install
- then run `git --version` again

### Step 3. Check whether Python is installed

Copy and paste this into Terminal:

```bash
python3 --version
```

Expected output:
- something like `Python 3.12.8`

If it prints a version:
- continue

If not:

1. install Homebrew if you do not already use it
2. then run:

```bash
brew install python
```

### Step 4. Check whether Tesseract is installed

Copy and paste this into Terminal:

```bash
tesseract --version
```

Expected output:
- first line looks something like `tesseract 5.5.0`

If it prints a version:
- continue

If not, install it:

```bash
brew install tesseract
```

### Step 5. Optional: install FFmpeg

Only needed for merge-video:

```bash
brew install ffmpeg
```

### Step 6. Choose a folder for the project

Example:

```bash
cd ~
mkdir -p Projects
cd Projects
```

### Step 7. Download the project

Replace `<GITHUB-URL>` with the real repository URL:

```bash
git clone <GITHUB-URL>
cd MarioKart8_lanplay_video_processor
```

Expected output:
- lines that mention `Cloning into ...`
- then no error when changing into the folder

### Step 8. Run the setup script

```bash
chmod +x ./scripts/setup_unix.sh
./scripts/setup_unix.sh
```

Expected output:
- install text
- then a runtime summary
- ideally including `Tesseract: OK`

### Step 9. Verify the install

```bash
.venv/bin/mk8-local-play --check
```

Expected output:
- `Tesseract: OK`
- `Child script Python: .../.venv/bin/python`

### Step 10. Add your videos

Put your files in:

- `Input_Videos/`

Open the folder in Finder:

```bash
open Input_Videos
```

Expected result:
- Finder opens the input folder

### Step 11. Run the full tool

```bash
.venv/bin/mk8-local-play --all
```

Expected output:
- extraction progress
- OCR progress
- final workbook path

### Step 12. Open the output folder

```bash
open Output_Results
```

Expected result:
- Finder opens the output folder

## Updating Later

If you already installed the project once and want the latest changes:

1. open a terminal in the project folder
2. run:

```bash
git pull
```

3. run the setup script again for your platform

Windows:

```powershell
.\scripts\setup_windows.ps1
```

Linux/macOS:

```bash
./scripts/setup_unix.sh
```

## If Something Does Not Work

First try these checks:

- `git --version`
- `python --version` or `python3 --version`
- `tesseract --version`
- `python -m mk8_local_play.main --check`

Then read:
- [README.md](./README.md)
- [docs/PROJECT_STRUCTURE.md](./docs/PROJECT_STRUCTURE.md)

If you still get stuck, collect:
- the command you ran
- the error message
- your platform
- whether Git, Python, and Tesseract showed version numbers
