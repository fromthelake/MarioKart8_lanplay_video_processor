### Mario Kart 8 Video-to-Text Analysis

This project allows you to analyze Mario Kart 8 race videos, extract relevant data such as player names and scores, and export them into an organized Excel file. The process is automated via a GUI for ease of use.
The video should be from Mario Kart 8 Local Play vertical split screen.

---

### Requirements
1. **Python Dependencies**:
   - Ensure you have all required Python packages installed by running:
     ```bash
     pip install -r requirements.txt
     ```
2. **Tesseract OCR**:
   - Install Tesseract OCR, as it is required for text recognition.
   - Download Tesseract from: [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - Update the `Extract_Text_From_Frames.py` script (line 18) with the installed Tesseract location:
     ```python
     pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
     ```
     *(The example above uses the default installation path on Windows.)*

3. **FFmpeg**:
   - FFmpeg is required for video processing.
   - Install FFmpeg and add its `bin` folder to your system's PATH.
     - **Windows**: Add `C:\\ffmpeg\\bin` to the PATH variable. [Detailed Instructions](https://ffmpeg.org/download.html)
     - **Linux/Mac**: Install via your package manager or from source.

---

### How to Use the Script
1. **Prepare Video Files**:
   - Place `.mkv` or `.mp4` videos in the `Input_Videos` folder.

2. **Run the GUI**:
   - Execute the `Main_RunMe.py` script to launch the GUI.

3. **Steps in the GUI**:
   - **Step 1 - Ensure Videos are in Input Folder**:
     - Verifies that videos are correctly placed in the `Input_Videos` folder.
   - **Optional - Merge Videos**:
     - Use this button to combine multiple video clips into a single file, if necessary.
     - Follow the prompts to select videos and specify the merged output file.
     - Only required if multiple clips should be treated as a single race pool.
   - **Step 2 - Analyze Videos and Find Races**:
     - Click this button to process the videos in the `Input_Videos` folder.
     - Extracted frames will be saved in the `.\Output_Results\Frames` folder.
   - **View Races Found**:
     - Opens the `Frames` folder to view extracted frames.
   - **Delete All Races Found**:
     - Removes all `.png` files from the `Frames` folder.
   - **Step 3 - Export Found Races into Excel**:
     - Processes extracted frames and exports race results to an Excel file.
     - The Excel file will be saved in the `.\Output_Results` folder as `Tournament_Results.xlsx`.
   - **Open Excel Scores**:
     - Opens the generated Excel file for review.

---

### Output Details
- The Excel file will include race results for all videos in the `Input_Videos` folder, provided the following frame types are present in the `.\Output_Results\Frames` folder:
  - `TrackName`
  - `RaceNumber`
  - `RaceScoreFrames`
  - `TotalScoreFrames`

---

### Additional Notes
- **Tesseract Configuration**:
  - Ensure the `pytesseract.pytesseract_cmd` in `Extract_Text_From_Frames.py` points to your Tesseract installation.

- **Supported Formats**:
  - Only `.mkv` and `.mp4` files are supported for analysis.

- **Output Structure**:
  - Extracted frames are stored in `.\Output_Results\Frames`.
  - The final results are compiled into `.\Output_Results\Tournament_Results.xlsx`.
"""
