Mariokart 8 Video to text.

Requirements:
Ensure you have all requirements installed, requirements can be found in requiremeents.txt or run the pip install command below.
pip install -r requirements.txt

In addition Tesseract needs to be installed on the PC:
Download from:
https://github.com/UB-Mannheim/tesseract/wiki

In addition install ffmpeg:

### Installing FFmpeg
1. **Windows**: Download and add `C:\ffmpeg\bin` to your PATH. [Detailed Instructions](https://ffmpeg.org/download.html)

Update the Extract_Text_From_Frames.py (line 18) script with the location tesseract is installed:
# Specify the path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
Example given above is the default location.

How to use the script.
Put the .mkv or .mp4 video in the folder "Input_Videos"

Run the Main_RunMe.py script which will give you a GUI.
Click Step 1 - wait to finish.
Images can be found in .\Output_Results\Frames folder.

Click Step 2 - wait to finish.
Excel file with playernames, scores etc, for all videos in "Input_Videos" folder can be found in Output_Results folder.

Note, the scores of a race will only be added if TrackName, RaceNumber, RaceScoreFrames and TotalScoreFrames are present in the .\Output_Results\Frames folder.
