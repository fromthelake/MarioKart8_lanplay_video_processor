import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import subprocess
import os
import glob
import sys

def select_video():
    # Get the absolute path of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to Extract_Frames_From_Video.py
    extract_frames_script = os.path.join(current_dir, "Extract_Frames_From_Video.py")

    try:
        subprocess.run([sys.executable, extract_frames_script], check=True)
        messagebox.showinfo("Success", "Video analyzed and races found.")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred while analyzing the video: {e}")
    except FileNotFoundError:
        messagebox.showerror("Error", f"Extract_Frames_From_Video.py script not found at {extract_frames_script}.")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")

def export_to_excel():
    # Get the absolute path of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to Extract_Text_From_Frames.py
    extract_text_script = os.path.join(current_dir, "Extract_Text_From_Frames.py")

    try:
        subprocess.run([sys.executable, extract_text_script], check=True)
        messagebox.showinfo("Success", "Races exported to Excel.")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred while exporting to Excel: {e}")
    except FileNotFoundError:
        messagebox.showerror("Error", f"Extract_Text_From_Frames.py script not found at {extract_text_script}.")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")

def open_excel_scores():
    # Get the absolute path of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the Excel file
    excel_path = os.path.join(current_dir, "Output_Results", "Tournament_Results.xlsx")

    if os.path.exists(excel_path):
        try:
            os.startfile(excel_path)
        except Exception as e:
            messagebox.showerror("Error", f"Unable to open the Excel file: {e}")
    else:
        messagebox.showwarning("Warning", "Please Perform Step 2 first.")

def open_frames_folder():
    # Get the absolute path of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the frames folder
    frames_folder = os.path.join(current_dir, "Output_Results", "Frames")

    if os.path.exists(frames_folder):
        try:
            os.startfile(frames_folder)
        except Exception as e:
            messagebox.showerror("Error", f"Unable to open the folder: {e}")
    else:
        messagebox.showwarning("Warning", "The frames folder does not exist.")

def clear_all_races_found():
    # Get the absolute path of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the frames folder
    frames_folder = os.path.join(current_dir, "Output_Results", "Frames")

    if os.path.exists(frames_folder):
        png_files = glob.glob(os.path.join(frames_folder, "*.png"))
        if png_files:
            for file in png_files:
                try:
                    os.remove(file)
                except Exception as e:
                    messagebox.showerror("Error", f"Unable to delete file {file}: {e}")
                    return
            messagebox.showinfo("Success", "All .png files have been deleted.")
        else:
            messagebox.showinfo("Info", "No .png files found to delete.")
    else:
        messagebox.showwarning("Warning", "The frames folder does not exist.")

def open_videos_folder():
    # Get the absolute path of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the Input_Videos folder
    videos_folder = os.path.join(current_dir, "Input_Videos")

    if os.path.exists(videos_folder):
        try:
            os.startfile(videos_folder)
        except Exception as e:
            messagebox.showerror("Error", f"Unable to open the folder: {e}")
    else:
        messagebox.showwarning("Warning", "The Input_Videos folder does not exist.")

def merge_videos():
    # Get the absolute path of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the Input_Videos folder
    input_videos_folder = os.path.join(current_dir, "Input_Videos")

    # Allow the user to select multiple videos to merge
    file_paths = filedialog.askopenfilenames(title="Select Videos to Merge", initialdir=input_videos_folder, filetypes=[("Video Files", "*.mp4;*.mkv;*.avi")])
    if not file_paths:
        return  # User cancelled selection

    # Ask for the output file name
    output_file = filedialog.asksaveasfilename(title="Save Merged Video As", defaultextension=".mp4", filetypes=[("MP4 Files", "*.mp4")])
    if not output_file:
        return  # User cancelled save dialog

    try:
        # Create a temporary file to list input files for ffmpeg
        temp_file = os.path.join(current_dir, "file_list.txt")
        with open(temp_file, "w") as f:
            for file_path in file_paths:
                f.write(f"file '{file_path}'\n")

        # Construct ffmpeg command
        ffmpeg_command = ["ffmpeg", "-f", "concat", "-safe", "0", "-i", temp_file, "-c", "copy", "-y", output_file]

        # Use ffmpeg to merge the videos
        subprocess.run(ffmpeg_command, check=True)
        messagebox.showinfo("Success", f"Videos merged successfully into {output_file}")

        # Clean up the temporary file
        os.remove(temp_file)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred while merging videos: {e}")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")

def exit_application():
    # Function to close the application
    root.destroy()

def main():
    global root
    # Create the main window
    root = tk.Tk()
    root.title("Mario Kart 8 Race Analysis")

    # Load Mario Kart 8 themed image
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mario_kart_image_path = os.path.join(current_dir, "GUI", "mariokart8_GUI_background_50.jpg")
    mario_kart_image = Image.open(mario_kart_image_path)  # Change the filename accordingly

    # Resize the image using LANCZOS algorithm
    mario_kart_image = mario_kart_image.resize((746, 420), Image.LANCZOS)
    mario_kart_image = ImageTk.PhotoImage(mario_kart_image)

    # Set window size
    root.geometry("746x420")

    # Create and place image label
    image_label = tk.Label(root, image=mario_kart_image)
    image_label.place(x=0, y=0)

    # Calculate vertical spacing
    top_spacing = 0.005
    button_height = (1 - top_spacing) / 8  # Eight buttons plus spacing

    # Create and place buttons
    open_videos_button = tk.Button(root, text="Step 1 - Ensure Videos are in Input Folder", command=open_videos_folder, font=("Helvetica", 16))
    open_videos_button.config(bg='#ffcc00', fg='#000000')  # Yellow button with black text
    open_videos_button.place(relx=0.5, rely=top_spacing + button_height * 0.5, anchor=tk.CENTER)

    merge_videos_button = tk.Button(root, text="Merge Videos", command=merge_videos, font=("Helvetica", 16))
    merge_videos_button.config(bg='#d3d3d3', fg='#000000')  # Light gray button with black text
    merge_videos_button.place(relx=0.5, rely=top_spacing + button_height * 1.4, anchor=tk.CENTER)

    merge_videos_note = tk.Label(root, text="(optional only needed for multiple clips which should be treated as a single Race Poule)", font=("Helvetica", 10), bg='#d3d3d3', fg='#000000')
    merge_videos_note.place(relx=0.5, rely=top_spacing + button_height * 2.0, anchor=tk.CENTER)

    analyze_button = tk.Button(root, text="Step 2 - Analyse Videos and Find Races", command=select_video, font=("Helvetica", 16))
    analyze_button.config(bg='#ffcc00', fg='#000000')  # Yellow button with black text
    analyze_button.place(relx=0.5, rely=top_spacing + button_height * 2.7, anchor=tk.CENTER)

    view_races_button = tk.Button(root, text="View Races Found", command=open_frames_folder, font=("Helvetica", 16))
    view_races_button.config(bg='#d3d3d3', fg='#000000')  # Light gray button with black text
    view_races_button.place(relx=0.5, rely=top_spacing + button_height * 3.6, anchor=tk.CENTER)

    clear_races_button = tk.Button(root, text="Delete All Races Found", command=clear_all_races_found, font=("Helvetica", 16))
    clear_races_button.config(bg='#ff4444', fg='#ffffff')  # Red button with white text
    clear_races_button.place(relx=0.5, rely=top_spacing + button_height * 4.5, anchor=tk.CENTER)

    export_button = tk.Button(root, text="Step 3 - Export Found Races into Excel", command=export_to_excel, font=("Helvetica", 16))
    export_button.config(bg='#ffcc00', fg='#000000')  # Yellow button with black text
    export_button.place(relx=0.5, rely=top_spacing + button_height * 5.4, anchor=tk.CENTER)

    open_excel_button = tk.Button(root, text="Open Excel Scores", command=open_excel_scores, font=("Helvetica", 16))
    open_excel_button.config(bg='#d3d3d3', fg='#000000')  # Light gray button with black text
    open_excel_button.place(relx=0.5, rely=top_spacing + button_height * 6.3, anchor=tk.CENTER)

    exit_button = tk.Button(root, text="Exit", command=exit_application, font=("Helvetica", 16))
    exit_button.config(bg='#ff4444', fg='#ffffff')  # Red button with white text
    exit_button.place(relx=0.5, rely=top_spacing + button_height * 7.2, anchor=tk.CENTER)

    # Run the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    main()
