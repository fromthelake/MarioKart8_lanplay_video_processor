import cv2
import numpy as np
import os
from glob import glob
import csv
import time
from PIL import Image, ImageEnhance, ImageFilter

# Record the start time
start_run_time = time.time()

print("Extract Frames Started - Calculating black borders")

# Global parameter for frame skip value
FRAME_SKIP = int(3 * 30)  # Skip 3 seconds (assuming 30 FPS)
LastTrackNameFrame = 0
LastRaceNumberFrame = 0
RaceCount = 1

def calculate_sum_intensity(gray_image):
    """Calculate the sum of pixel intensities for rows and columns."""
    sum_row_intensity = np.sum(gray_image, axis=1)
    sum_col_intensity = np.sum(gray_image, axis=0)
    return sum_row_intensity, sum_col_intensity

def find_borders(sum_row_intensity, sum_col_intensity, threshold=25000):
    """Find the borders of the active game area based on intensity sums."""
    top = next((i for i, val in enumerate(sum_row_intensity) if val > threshold), 0)
    bottom = next((i for i, val in enumerate(reversed(sum_row_intensity)) if val > threshold), 0)
    bottom = len(sum_row_intensity) - bottom
    left = next((i for i, val in enumerate(sum_col_intensity) if val > threshold), 0)
    right = next((i for i, val in enumerate(reversed(sum_col_intensity)) if val > threshold), 0)
    right = len(sum_col_intensity) - right
    return top, left, bottom, right

def determine_scaling(image):
    """Determine the scaling factors and crop dimensions for the image."""
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sum_row_intensity, sum_col_intensity = calculate_sum_intensity(gray_frame)
    top, left, bottom, right = find_borders(sum_row_intensity, sum_col_intensity, threshold=15000)

    height, width = image.shape[:2]
    crop_width = right - left
    crop_height = bottom - top
    scale_x = 1280 / crop_width
    scale_y = 720 / crop_height

    return scale_x, scale_y, left, top, crop_width, crop_height

def load_videos_from_folder(folder_path):
    """Load video file paths from the specified folder."""
    video_extensions = ["*.mp4", "*.mkv", "*.mov", "*.avi", "*.webm"]
    video_paths = []
    for extension in video_extensions:
        video_paths.extend(glob(os.path.join(folder_path, extension)))
    return video_paths

def crop_and_upscale_image(image, left, top, crop_width, crop_height, target_width, target_height):
    """Crop the image to the detected borders and upscale to the target size."""
    cropped_image = image[top:top + crop_height, left:left + crop_width]
    upscaled_image = cv2.resize(cropped_image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return upscaled_image


def preprocess_roi(roi, process_type):
    """Preprocess the Region of Interest (ROI) based on the process type."""
    if process_type == 0:
        _, binary_section = cv2.threshold(roi, 180, 255, cv2.THRESH_BINARY)
        modified_binary_section = binary_section.copy()
        step_height = max(1, roi.shape[0] // 12)
        for i in range(12):
            sub_region_y_start = i * step_height
            sub_region_y_end = min((i + 1) * step_height, roi.shape[0])
            sub_region = binary_section[sub_region_y_start:sub_region_y_end, :]
            if sub_region.size == 0:
                continue
            black_pixels = np.count_nonzero(sub_region == 0)
            white_pixels = np.count_nonzero(sub_region == 255)
            if white_pixels > black_pixels:
                _, binary_section_sub = cv2.threshold(sub_region, 120, 255, cv2.THRESH_BINARY)
                sub_region_copy = binary_section_sub.copy()
                kernel = np.ones((2, 1), np.uint8)
                sub_region_copy = cv2.dilate(sub_region_copy, kernel, iterations=1)
                sub_region_copy = cv2.bitwise_not(sub_region_copy)
                modified_binary_section[sub_region_y_start:sub_region_y_end, :] = sub_region_copy
            else:
                kernel = np.ones((2, 1), np.uint8)
                dilated_section = cv2.dilate(sub_region, kernel, iterations=1)
                modified_binary_section[sub_region_y_start:sub_region_y_end, :] = dilated_section
    else:
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        if len(blurred.shape) == 3 and blurred.shape[2] == 3:
            gray_blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        else:
            gray_blurred = blurred
        thresholded = cv2.adaptiveThreshold(gray_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        modified_binary_section = cv2.filter2D(thresholded, -1, kernel)
    return modified_binary_section

def frame_to_timecode(frame_number, fps):
    """Convert frame number to timecode in HH:MM:SS format."""
    seconds = frame_number / fps
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def process_frame(frame, frame_number, video_path, templates, fps, csv_writer, scale_x, scale_y, left, top, crop_width, crop_height):
    """Process a single video frame and apply template matching."""
    global LastTrackNameFrame
    global LastRaceNumberFrame
    global RaceCount
    global cap
    global FRAME_SKIP

    # Crop and upscale the image using the calculated scaling factors
    target_width, target_height = 1280, 720
    upscaled_image = crop_and_upscale_image(frame, left, top, crop_width, crop_height, target_width, target_height)


    # Convert the image to grayscale
    gray_image = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)

    for i in range(3):
        if i == 0:
            # Define the ROI for RaceScore
            roi_x, roi_y, roi_width, roi_height = 315, 57, 52, 610
            TargetColumn = "Score"
        elif i == 1:
            # Define the ROI for TrackName
            roi_x, roi_y, roi_width, roi_height = 141, 607, 183, 101
            TargetColumn = "TrackName"
        else:
            # Define the ROI for RaceNumber
            roi_x, roi_y, roi_width, roi_height = 640, 590, 144, 48
            TargetColumn = "RaceNumber"

        # Extend ROI by 25 pixels in each direction
        roi_x = max(roi_x - 25, 0)
        roi_y = max(roi_y - 25, 0)
        roi_width = min(roi_width + 50, gray_image.shape[1] - roi_x)
        roi_height = min(roi_height + 50, gray_image.shape[0] - roi_y)
        roi = gray_image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

        # Preprocess the ROI
        processed_roi = preprocess_roi(roi, i)

        # Convert frame number to timecode
        timecode = frame_to_timecode(frame_number, fps)

        # Skip the frame if 97% or more of the pixels in the ROI are black
        black_pixel_percentage = np.mean(processed_roi == 0)
        if black_pixel_percentage >= 0.97:
            csv_writer.writerow([os.path.basename(video_path), "Score", frame_number, 0, timecode])
            #we need to ensure all templates are checked before quitting for black screens.
            if i == 2:
                return 0
            else:
                continue

        template_binary, alpha_mask = templates[i]
        if processed_roi.shape[0] < template_binary.shape[0] or processed_roi.shape[1] < template_binary.shape[1]:
            processed_roi = cv2.resize(processed_roi, (max(template_binary.shape[1], processed_roi.shape[1]),
                                                       max(template_binary.shape[0], processed_roi.shape[0])),
                                       interpolation=cv2.INTER_LINEAR)

        res = cv2.matchTemplate(processed_roi, template_binary, cv2.TM_CCOEFF_NORMED, mask=alpha_mask)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        if i == 0 and max_val > 0.3 and not np.isinf(max_val):
            #"""RaceScore Frame found now we want to analyse the details."""
            current_frame = frame_number
            start_frame = frame_number - int(3 * fps)
            end_frame = frame_number + int(13 * fps)
            RaceScoreFrame = 0
            TotalScoreFrame = 0
            player12 = 0
            Check_player_12 = 0

            frame_number = start_frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            #print(f"Detail Analyse StartFrame:{start_frame} EndFrame:{end_frame}")
            while frame_number < end_frame:
                # Capture frame-by-frame
                ret, frame = cap.read()
                # Check if frame was read successfully
                if not ret:
                    break

                timecode = frame_to_timecode(frame_number, fps)
                upscaled_image = crop_and_upscale_image(frame, left, top, crop_width, crop_height, target_width, target_height)
                gray_image = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)
                roi_x, roi_y, roi_width, roi_height = 315, 57, 52, 610
                roi_x = max(roi_x - 25, 0)
                roi_y = max(roi_y - 25, 0)
                roi_width = min(roi_width + 50, gray_image.shape[1] - roi_x)
                roi_height = min(roi_height + 50, gray_image.shape[0] - roi_y)
                roi = gray_image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
                processed_roi = preprocess_roi(roi, 0)
                black_pixel_percentage = np.mean(processed_roi == 0)
                if black_pixel_percentage >= 0.97:
                    frame_number += 1
                    #print(f"Video: {os.path.basename(video_path)}, Score {RaceCount:03} at Timecode: {timecode} and Frame:{frame_number}, Max Value: 0")
                    csv_writer.writerow([os.path.basename(video_path), "Score", frame_number, 0, timecode])
                    if RaceScoreFrame != 0:
                        TotalScoreFrame = frame_number - int(2.7 * fps)
                        #print(f"TotalScoreFrame:{TotalScoreFrame}")
                        break
                    continue

                if processed_roi.shape[0] < template_binary.shape[0] or processed_roi.shape[1] < template_binary.shape[1]:
                    processed_roi = cv2.resize(processed_roi, (max(template_binary.shape[1], processed_roi.shape[1]),
                                                               max(template_binary.shape[0], processed_roi.shape[0])),
                                               interpolation=cv2.INTER_LINEAR)
                res = cv2.matchTemplate(processed_roi, template_binary, cv2.TM_CCOEFF_NORMED, mask=alpha_mask)
                _, max_val, _, _ = cv2.minMaxLoc(res)

                #print(f"Video: {os.path.basename(video_path)}, Score {RaceCount:03} at Timecode: {timecode} and Frame:{frame_number}, Max Value: {max_val}")
                csv_writer.writerow([os.path.basename(video_path), "Score", frame_number, max_val, timecode])

                if max_val > 0.3 and not np.isinf(max_val) and RaceScoreFrame == 0:
                    RaceScoreFrame = frame_number + int(0.6 * fps)
                    Check_player_12 = 1
                    #print("ONE!!")
                    #frame_number += int(3.9 * fps)
                    #cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    #print(f"RaceScoreFrame:{RaceScoreFrame}")
                    continue

                if max_val > 0.3 and not np.isinf(max_val) and Check_player_12 == 1:
                    #print("TWO!!")
                    # #if 12th we have an overlap problem, continue searching the frame with the highest score.
                    #template match 12th
                    upscaled_image2 = crop_and_upscale_image(frame, left, top, crop_width, crop_height, target_width, target_height)
                    gray_image2 = cv2.cvtColor(upscaled_image2, cv2.COLOR_BGR2GRAY)
                    roi_x2, roi_y2, roi_width2, roi_height2 = 338, 657, 601, 39
                    roi_x2 = max(roi_x2 - 25, 0)
                    roi_y2 = max(roi_y2 - 25, 0)
                    roi_width2 = min(roi_width2 + 50, gray_image2.shape[1] - roi_x2)
                    roi_height2 = min(roi_height2 + 50, gray_image2.shape[0] - roi_y2)
                    roi2 = gray_image2[roi_y2:roi_y2 + roi_height2, roi_x2:roi_x2 + roi_width2]
                    processed_roi2 = preprocess_roi(roi2, 0)



                    #check 12th row template
                    template_binary2, alpha_mask2 = templates[3]

                    if processed_roi2.shape[0] < template_binary2.shape[0] or processed_roi2.shape[1] < \
                            template_binary2.shape[1]:
                        processed_roi2 = cv2.resize(processed_roi2,
                                                   (max(template_binary2.shape[1], processed_roi2.shape[1]),
                                                    max(template_binary2.shape[0], processed_roi2.shape[0])),
                                                   interpolation=cv2.INTER_LINEAR)
                    res = cv2.matchTemplate(processed_roi2, template_binary2, cv2.TM_CCOEFF_NORMED, mask=alpha_mask2)
                    _, max_val2, _, _ = cv2.minMaxLoc(res)
                    #print(f"12th Video: {os.path.basename(video_path)}, Score {RaceCount:03} at Timecode: {timecode} and Frame:{frame_number}, Max Value2: {max_val2}")
                    #12thplayer finished detected
                    if max_val2 > 0.4 and not np.isinf(max_val2):
                        player12 = 1

                    if player12 == 1 and max_val2 < 0.1:
                        RaceScoreFrame = frame_number + int(16)
                        frame_number += int(3.9 * fps)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                        Check_player_12 = 2
                        continue



                    #frame_number += int(3.9 * fps)
                    #cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    #print(f"RaceScoreFrame:{RaceScoreFrame}")

                if max_val <= 0 and RaceScoreFrame != 0:
                    TotalScoreFrame = frame_number - int(2.7 * fps)
                    #print(f"TotalScoreFrame:{TotalScoreFrame}")
                    break

                frame_number += 1

            # Detail Race Score Frame found now we load the frames and then save them
            timecode = frame_to_timecode(RaceScoreFrame, fps)
            #print(f"Video: {os.path.basename(video_path)}, RaceScoreFrame {RaceCount:03} at Timecode: {timecode}")

            cap.set(cv2.CAP_PROP_POS_FRAMES, RaceScoreFrame)
            ret, frame = cap.read()
            if not ret:
                break
            upscaled_image = crop_and_upscale_image(frame, left, top, crop_width, crop_height, target_width, target_height)

            # When the video resolution is low, we need additional preprocessing to sharpen the text.
            # When the video resolution is low, we need additional preprocessing to sharpen the text.
            if scale_x > 1.3 and scale_y >= 1.3:
                # Ensure upscaled_image is a PIL Image
                if isinstance(upscaled_image, np.ndarray):
                    upscaled_image = Image.fromarray(upscaled_image)

                # Increase contrast
                contrast_enhancer = ImageEnhance.Contrast(upscaled_image)
                high_contrast_image = contrast_enhancer.enhance(1.70)

                # Sharpen the image
                sharpness_enhancer = ImageEnhance.Sharpness(high_contrast_image)
                sharpened_image = sharpness_enhancer.enhance(1.23)
                # Convert the PIL image to a numpy array
                upscaled_image = np.array(sharpened_image)

            script_dir = os.path.dirname(__file__)  # Directory of the script
            output_folder = os.path.join(script_dir, 'Output_Results', 'Frames')
            os.makedirs(output_folder, exist_ok=True)
            frame_filename = os.path.join(output_folder,
                                          f"{os.path.splitext(os.path.basename(video_path))[0]}+Race_{RaceCount:03}+2RaceScore.png")
            cv2.imwrite(frame_filename, upscaled_image)

            # Detail Total Score Frame found now we load the frames and then save them
            timecode = frame_to_timecode(TotalScoreFrame, fps)
            print(f"Video: {os.path.basename(video_path)}, TotalScoreFrame {RaceCount:03} at Timecode: {timecode}")

            cap.set(cv2.CAP_PROP_POS_FRAMES, TotalScoreFrame)
            ret, frame = cap.read()
            if not ret:
                break
            upscaled_image = crop_and_upscale_image(frame, left, top, crop_width, crop_height, target_width, target_height)

            # When the video resolution is low, we need additional preprocessing to sharpen the text.
            if scale_x > 1.3 and scale_y >= 1.3:
                # Ensure upscaled_image is a PIL Image
                if isinstance(upscaled_image, np.ndarray):
                    upscaled_image = Image.fromarray(upscaled_image)

                # Increase contrast
                contrast_enhancer = ImageEnhance.Contrast(upscaled_image)
                high_contrast_image = contrast_enhancer.enhance(1.70)

                # Sharpen the image
                sharpness_enhancer = ImageEnhance.Sharpness(high_contrast_image)
                sharpened_image = sharpness_enhancer.enhance(1.23)
                # Convert the PIL image to a numpy array
                upscaled_image = np.array(sharpened_image)



            frame_filename = os.path.join(output_folder,
                                          f"{os.path.splitext(os.path.basename(video_path))[0]}+Race_{RaceCount:03}+3TotalScore.png")
            cv2.imwrite(frame_filename, upscaled_image)

            #Set back to the last frame we did before starting the detail analyses
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                break
            frame_number = current_frame

            csv_writer.writerow([os.path.basename(video_path), "Score", frame_number, max_val, timecode])

            #we can skip 20 seconds knowing a new game will not start within 20 seconds from end score screen.
            frames_to_skip = int(fps * 20)
            RaceCount += 1
            return frames_to_skip
        elif i == 1 and max_val > 0.6 and not np.isinf(max_val):
            if LastTrackNameFrame < max(1, frame_number - int(fps * 20)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number + int(fps * 1))
                ret, frame = cap.read()
                if not ret:
                    break
                upscaled_image = crop_and_upscale_image(frame, left, top, crop_width, crop_height, target_width, target_height)
                LastTrackNameFrame = frame_number
                print(f"Video: {os.path.basename(video_path)}, TrackName {RaceCount:03} Timecode: {timecode}, Max Value: {max_val}")
                script_dir = os.path.dirname(__file__)  # Directory of the script
                output_folder = os.path.join(script_dir, 'Output_Results', 'Frames')
                os.makedirs(output_folder, exist_ok=True)
                frame_filename = os.path.join(output_folder,
                                              f"{os.path.splitext(os.path.basename(video_path))[0]}+Race_{RaceCount:03}+0TrackName.png")
                cv2.imwrite(frame_filename, upscaled_image)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                if not ret:
                    break
            csv_writer.writerow([os.path.basename(video_path), "TrackName", frame_number, max_val, timecode])
            return 0
        elif i == 2 and max_val > 0.6 and not np.isinf(max_val):
            if LastRaceNumberFrame < max(1, frame_number - int(fps * 20)):
                LastRaceNumberFrame = frame_number
                print(f"Video: {os.path.basename(video_path)}, RaceNumber {RaceCount:03} at Timecode: {timecode}, Max Value: {max_val}")
                script_dir = os.path.dirname(__file__)  # Directory of the script
                output_folder = os.path.join(script_dir, 'Output_Results', 'Frames')

                os.makedirs(output_folder, exist_ok=True)
                frame_filename = os.path.join(output_folder,
                                              f"{os.path.splitext(os.path.basename(video_path))[0]}+Race_{RaceCount:03}+1RaceNumber.png")
                cv2.imwrite(frame_filename, upscaled_image)
            csv_writer.writerow([os.path.basename(video_path), "RaceNumber", frame_number, max_val, timecode])
            frames_to_skip = int(fps * 60)
            return frames_to_skip

        if i == 0:
            csv_writer.writerow([os.path.basename(video_path), "Score", frame_number, max_val, timecode])
        elif i == 1:
            csv_writer.writerow([os.path.basename(video_path), "TrackName", frame_number, max_val, timecode])
        else:
            csv_writer.writerow([os.path.basename(video_path), "RaceNumber", frame_number, max_val, timecode])
    return 0

def main():
    """Main function to process videos and apply template matching."""

    script_dir = os.path.dirname(__file__)  # Directory of the script
    folder_path = os.path.join(script_dir, 'Input_Videos')

    template_paths = [
        (os.path.join(script_dir, 'Find_Templates', 'Score_template.png'), None),
        (os.path.join(script_dir, 'Find_Templates', 'Trackname_template.png'), None),
        (os.path.join(script_dir, 'Find_Templates', 'Race_template.png'), None),
        (os.path.join(script_dir, 'Find_Templates', '12th_pos_template.png'), None)
    ]

    templates = []
    for template_path, _ in template_paths:
        template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
        if template is None:
            print(f"Error: Template image at '{template_path}' could not be loaded.")
            return
        if len(template.shape) == 3 and template.shape[2] == 4:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGRA2GRAY)
            _, alpha_mask = cv2.threshold(template[:, :, 3], 0, 255, cv2.THRESH_BINARY)
            _, template_binary = cv2.threshold(template_gray, 180, 255, cv2.THRESH_BINARY)
        elif len(template.shape) == 3 and template.shape[2] == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            _, template_binary = cv2.threshold(template_gray, 180, 255, cv2.THRESH_BINARY)
            alpha_mask = np.ones(template_binary.shape, dtype=np.uint8) * 255
        elif len(template.shape) == 2:
            template_binary = template
            alpha_mask = np.ones(template_binary.shape, dtype=np.uint8) * 255
        else:
            print(f"Error: Template image at '{template_path}' has an unexpected number of channels.")
            return
        templates.append((template_binary, alpha_mask))

    csv_output_path = os.path.join(script_dir, 'Output_Results', 'Debug', 'debug_max_val.csv')
    video_paths = load_videos_from_folder(folder_path)
    if not video_paths:
        print("No videos found in the specified folder. Exiting.")
        return

    with open(csv_output_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=';')
        csv_writer.writerow(["Video", "Target", "Frame Number", "Max Value", "Timecode"])

        for video_path in video_paths:
            global LastTrackNameFrame
            global LastRaceNumberFrame
            global RaceCount
            global cap
            global FRAME_SKIP

            LastTrackNameFrame = 0
            LastRaceNumberFrame = 0
            RaceCount = 1
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                continue
            #Determine the FPS
            fps = cap.get(cv2.CAP_PROP_FPS)
            FRAME_SKIP = int(3 * int(fps))

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_frames = np.linspace(0, total_frames - 1, 19).astype(int)
            scales = []

            for frame_num in sample_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    continue
                scale_x, scale_y, left, top, crop_width, crop_height = determine_scaling(frame)
                scales.append((scale_x, scale_y, left, top, crop_width, crop_height))

            if not scales:
                print(f"Error: No valid frames found for scaling in video {video_path}.")
                continue

            median_scale_x = np.median([s[0] for s in scales])
            median_scale_y = np.median([s[1] for s in scales])
            median_left = int(np.median([s[2] for s in scales]))
            median_top = int(np.median([s[3] for s in scales]))
            median_crop_width = int(np.median([s[4] for s in scales]))
            median_crop_height = int(np.median([s[5] for s in scales]))

            print(f"Median scale_x: {median_scale_x}, Median scale_y: {median_scale_y}")
            print(f"Median left: {median_left}, Median top: {median_top}")
            print(f"Median crop_width: {median_crop_width}, Median crop_height: {median_crop_height}")

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frames_to_skip = process_frame(frame, frame_count, video_path, templates, fps, csv_writer,
                                               median_scale_x, median_scale_y, median_left, median_top, median_crop_width, median_crop_height)

                frame_count += frames_to_skip
                frame_count += FRAME_SKIP
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            cap.release()

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_run_time

    # Convert elapsed time to minutes and seconds
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    # Print the elapsed time in mm:ss format
    print(f"Runtime was: {minutes:02}:{seconds:02}")

if __name__ == "__main__":
    main()
