# Mario Kart 8 LAN Play Video Processor

Short setup guide for Windows.

GitHub:
- https://github.com/fromthelake/MarioKart8_lanplay_video_processor

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

```powershell
git --version
```

If it works:
- continue to Step 3

If it fails:
- download and install Git for Windows:
  - https://git-scm.com/download/win
- open a new PowerShell window
- run `git --version` again

## Step 3. Check Python

Run:

```powershell
python --version
```

If that fails, run:

```powershell
py --version
```

If one of them works:
- continue to Step 4

If both fail:
- download and install Python:
  - https://www.python.org/downloads/windows/
- during install, enable `Add Python to PATH` if shown
- open a new PowerShell window
- run `python --version` again

## Step 4. Check Tesseract OCR

Run:

```powershell
tesseract --version
```

If it works:
- continue to Step 5

If it fails:
- install Tesseract for Windows:
  - https://ub-mannheim.github.io/Tesseract_Dokumentation/Tesseract_Doku_Windows.html
- open a new PowerShell window
- run `tesseract --version` again

## Step 5. Download the project

Run:

```powershell
git clone https://github.com/fromthelake/MarioKart8_lanplay_video_processor
cd MarioKart8_lanplay_video_processor
```

## Step 6. Run setup

Run:

```powershell
.\scripts\setup_windows.ps1
```

If setup succeeds:
- continue to Step 7

If setup fails:
- read the error shown in PowerShell
- fix the missing dependency
- run `./scripts/setup_windows.ps1` again

## Step 7. Run the environment check

Run:

```powershell
.\.venv\Scripts\mk8-local-play.exe --check
```

If the check succeeds:
- continue to Step 8

If the check says Tesseract is missing:
- install Tesseract from:
  - https://ub-mannheim.github.io/Tesseract_Dokumentation/Tesseract_Doku_Windows.html
- run the check again

## Step 8. Add your videos

Put your video files in:

```text
Input_Videos
```

## Step 9. Run the tool

Process everything in `Input_Videos`:

```powershell
.\.venv\Scripts\mk8-local-play.exe --all
```

Process only the current selected input set:

```powershell
.\.venv\Scripts\mk8-local-play.exe --selection
```

## Output

Results are written to:

```text
Output_Results
```

## Optional commands

Run extraction only:

```powershell
.\.venv\Scripts\mk8-local-play.exe --extract
```

Run OCR/export only:

```powershell
.\.venv\Scripts\mk8-local-play.exe --ocr
```

Run one video only:

```powershell
.\.venv\Scripts\mk8-local-play.exe --all --video Demo_CaptureCard_Race.mp4
```

## If you want more detail

For the longer beginner guide, read:
- [BEGINNER_SETUP.md](./BEGINNER_SETUP.md)

## Technical Reference

If you want the pipeline, templates, ROIs, and metadata documented for development or reproduction, read:
- [docs/TECHNICAL_PIPELINE.md](./docs/TECHNICAL_PIPELINE.md)
