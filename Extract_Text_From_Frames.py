import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import pytesseract
import pandas as pd
from typing import List, Dict, Tuple
import logging
from PIL import Image
import time
import sys
import difflib
import openpyxl
import textdistance
import re
from jellyfish import soundex
from collections import defaultdict, Counter

# Specify the path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Record the start time
start_run_time = time.time()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def apply_threshold(image: np.ndarray, threshold: int = 205) -> np.ndarray:
    """Apply a binary threshold to the image."""
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def apply_inversion(image: np.ndarray) -> np.ndarray:
    """Invert the binary image."""
    return cv2.bitwise_not(image)

def preprocess_image_v_channel(image: np.ndarray, threshold_value: int = 205) -> np.ndarray:
    """Preprocess the image using the V channel from the HSV color space."""
    if len(image.shape) == 2:  # If the image is grayscale, convert it to BGR
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv_image)
    _, binary_v = cv2.threshold(v, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_v

def get_race_points(position: int, num_players: int) -> int:
    """Return the points based on position and number of players."""
    # Define point tables for different player counts
    points_table = {
        12: [15, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        11: [13, 11, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        10: [12, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        9: [11, 9, 8, 6, 5, 4, 3, 2, 1, 0],
        8: [10, 8, 6, 5, 4, 3, 2, 1, 0],
    }

    # Default to 12 players if not explicitly defined
    if num_players not in points_table:
        num_players = 12

    # Get the points for the given position
    if 1 <= position <= len(points_table[num_players]):
        return points_table[num_players][position - 1]
    else:
        return 0  # No points for invalid positions

def crop_and_process_image(frame: np.ndarray, coordinates: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                           image_type: str) -> List[np.ndarray]:
    """Crop and process specific regions of an image based on coordinates and image type."""
    cropped_images = []
    for (x1, y1), (x2, y2) in coordinates:
        section_img = frame[y1:y2, x1:x2]
        threshold_value = 205
        binary_section = preprocess_image_v_channel(section_img, threshold_value)
        black_pixels = np.count_nonzero(binary_section == 0)
        white_pixels = np.count_nonzero(binary_section == 255)

        if white_pixels > black_pixels:
            inverted_section = cv2.bitwise_not(binary_section)
            section_img = Image.fromarray(inverted_section)
        else:
            section_img = Image.fromarray(binary_section)

        # Apply dilation if the image_type is "race_points" or "total_points"
        if image_type in ["race_points", "total_points"]:
            # Convert the PIL image to a NumPy array (OpenCV image)
            section_img_np = np.array(section_img)
            if len(section_img_np.shape) == 3 and section_img_np.shape[2] == 3:
                # Convert to grayscale if it is an RGB image
                gray_image = cv2.cvtColor(section_img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = section_img_np

            # Define the kernel for dilation
            kernel = np.ones((2, 2), np.uint8)
            # Apply dilation
            dilated_image = cv2.dilate(gray_image, kernel, iterations=1)
            # Convert the dilated image back to a PIL image
            section_img = Image.fromarray(cv2.cvtColor(dilated_image, cv2.COLOR_GRAY2RGB))

        cropped_images.append(np.array(section_img))
    return cropped_images

def process_image(image_path: str) -> np.ndarray:
    """Process an image by cropping and modifying specified regions based on coordinates."""
    coordinates = {
        "player_position": [
            ((310, 52), (371, 96)), ((310, 104), (371, 148)),
            ((310, 156), (371, 200)), ((310, 208), (371, 252)),
            ((310, 260), (371, 304)), ((310, 312), (371, 356)),
            ((310, 364), (371, 408)), ((310, 416), (371, 460)),
            ((310, 468), (371, 512)), ((310, 520), (371, 564)),
            ((310, 572), (371, 617)), ((310, 624), (371, 669))
        ],
        "player_name": [
            ((428, 52), (620, 96)), ((428, 104), (620, 148)),
            ((428, 156), (620, 200)), ((428, 208), (620, 252)),
            ((428, 260), (620, 304)), ((428, 312), (620, 356)),
            ((428, 364), (620, 408)), ((428, 416), (620, 460)),
            ((428, 468), (620, 512)), ((428, 520), (620, 564)),
            ((428, 572), (620, 617)), ((428, 624), (620, 669))
        ],
        "race_points": [
            ((825, 52), (861, 96)), ((825, 104), (861, 148)),
            ((825, 156), (861, 200)), ((825, 208), (861, 252)),
            ((825, 260), (861, 304)), ((825, 312), (861, 356)),
            ((825, 364), (861, 408)), ((825, 416), (861, 460)),
            ((825, 468), (861, 512)), ((825, 520), (861, 564)),
            ((825, 572), (861, 617)), ((825, 624), (861, 669))
        ],
        "total_points": [
            ((910, 52), (973, 96)), ((910, 104), (973, 148)),
            ((910, 156), (973, 200)), ((910, 208), (973, 252)),
            ((910, 260), (973, 304)), ((910, 312), (973, 356)),
            ((910, 364), (973, 408)), ((910, 416), (973, 460)),
            ((910, 468), (973, 512)), ((910, 520), (973, 564)),
            ((910, 572), (973, 617)), ((910, 624), (973, 669))
        ]
    }

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Preprocess the image using the V channel
    processed_image = preprocess_image_v_channel(image)

    # Convert to 3-channel if processed_image is single-channel
    if len(processed_image.shape) == 2:
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)

    # Process each region type and its coordinates
    for region_type, coord_list in coordinates.items():
        rois = crop_and_process_image(processed_image, coord_list, region_type)
        for roi, ((x1, y1), (x2, y2)) in zip(rois, coord_list):
            # Ensure roi is 3-channel
            if len(roi.shape) == 2:
                roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            processed_image[y1:y2, x1:x2] = roi
    return processed_image

def is_white_box(image: Image.Image, top_left: Tuple[int, int], box_size: Tuple[int, int] = (3, 2)) -> bool:
    """Determine if a 3x2 pixel box is white based on RGB values."""
    x, y = top_left
    width, height = box_size
    white_pixels = 0
    total_pixels = width * height

    for i in range(width):
        for j in range(height):
            r, g, b = image.getpixel((x + i, y + j))
            if r > 180 and g > 180 and b > 180:
                white_pixels += 1

    return white_pixels >= total_pixels / 2

def identify_digit(image: Image.Image, box_top_left: Tuple[int, int], red_pixels: Dict[str, Tuple[int, int]]) -> int:
    """Identify the digit in a cropped region of an image based on pixel analysis."""
    white_pixels = {label: is_white_box(image, (box_top_left[0] + x, box_top_left[1] + y))
                    for label, (x, y) in red_pixels.items()}

    digit_patterns = [
        (8, {"top_middle", "left_middle", "right_middle", "center", "right_bottom", "left_bottom", "middle_bottom_edge"}),
        (0, {"top_middle", "left_middle", "right_middle", "left_bottom", "right_bottom", "middle_bottom_edge"}),
        (6, {"top_middle", "left_middle", "center", "right_bottom", "left_bottom", "middle_bottom_edge"}),
        (9, {"top_middle", "left_middle", "right_middle", "center", "right_bottom", "middle_bottom_edge"}),
        (2, {"top_middle", "right_middle", "center", "left_bottom", "middle_bottom_edge"}),
        (3, {"top_middle", "right_middle", "center", "right_bottom", "middle_bottom_edge"}),
        (5, {"top_middle", "left_middle", "center", "right_bottom", "middle_bottom_edge"}),
        (4, {"right_middle", "left_middle", "center", "right_bottom"}),
        (7, {"top_middle", "right_middle", "right_bottom"}),
        (1, {"middle_middle", "middle_bottom"})
    ]

    for digit, pattern in digit_patterns:
        if all(white_pixels.get(label, False) for label in pattern):
            return digit

    return -1

def detect_digits_in_image(image: Image.Image, start_coords: List[Tuple[int, int]], row_offset: int,
                           box_dims: Tuple[int, int], red_pixels: Dict[str, Tuple[int, int]],
                           num_rows: int, boxes_per_row: int) -> List[str]:
    """Detect digits in an image based on given coordinates and draw red pixels."""
    coordinate_set = []
    draw = ImageDraw.Draw(image)

    for i in range(num_rows):
        y_offset = i * row_offset
        row_number = ""

        for j in range(boxes_per_row):
            start_x, start_y = start_coords[j]
            top_left = (start_x, start_y + y_offset)
            digit = identify_digit(image, top_left, red_pixels)
            if digit != -1:
                row_number += str(digit)
            # Draw a red rectangle at the center of the digit box
            for label, (x, y) in red_pixels.items():
                rect_top_left = (top_left[0] + x, top_left[1] + y)
                rect_bottom_right = (rect_top_left[0] + 3, rect_top_left[1] + 2)
                draw.rectangle([rect_top_left, rect_bottom_right], outline='red', fill='red')

        coordinate_set.append(row_number)

    return coordinate_set

def scale_coords(coords, scale_factor):
    """Scale coordinates by a given factor."""
    return [(x * scale_factor, y * scale_factor) for x, y in coords]

def scale_pixel_positions(pixels, scale_factor):
    """Scale pixel positions by a given factor."""
    return {label: (x * scale_factor, y * scale_factor) for label, (x, y) in pixels.items()}


def extract_text_with_confidence(image_path: str, coordinates: Dict[str, List[Tuple[Tuple[int, int], Tuple[int, int]]]],
                                 lang: str, config: str) -> Tuple[Dict[str, List[str]], List[int]]:
    """Extract text and confidence scores from image ROIs using Tesseract OCR."""
    extracted_text = {}
    confidence_scores = []
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    for region_type, coord_list in coordinates.items():
        region_text = []
        region_confidence = []
        for (x1, y1), (x2, y2) in coord_list:
            roi = image[y1:y2, x1:x2]
            data = pytesseract.image_to_data(roi, lang=lang, config=config, output_type=pytesseract.Output.DICT)

            # Combine all words with a space
            words_in_roi = []
            confidences_in_roi = []
            for idx, conf in enumerate(data["conf"]):
                if conf > 0 and data["text"][idx].strip():  # Only valid words
                    words_in_roi.append(data["text"][idx].strip())
                    confidences_in_roi.append(conf)

            # Combine words into a single string and calculate average confidence
            combined_text = " ".join(words_in_roi)
            average_confidence = sum(confidences_in_roi) // len(confidences_in_roi) if confidences_in_roi else 0

            region_text.append(combined_text)
            region_confidence.append(average_confidence)

        extracted_text[region_type] = region_text  # List of combined texts for each ROI
        confidence_scores.extend(region_confidence)

    return extracted_text, confidence_scores


# List of official track names
tracks_list = [
    (1, "Mario Kart Stadium", "Mario Kart-Stadion", 1, "Mushroom Cup"),
    (2, "Water Park", "Waterpretpark", 1, "Mushroom Cup"),
    (3, "Sweet Sweet Canyon", "Zoetekauwcanyon", 1, "Mushroom Cup"),
    (4, "Thwomp Ruins", "Thwomps Tempel", 1, "Mushroom Cup"),
    (5, "Mario Circuit", "Mario's Circuit", 2, "Flower Cup"),
    (6, "Toad Harbour", "Toad-Baai", 2, "Flower Cup"),
    (7, "Twisted Mansion", "Boo's Duizelhuis", 2, "Flower Cup"),
    (8, "Shy Guy Falls", "Shy Guys Kristalmijn", 2, "Flower Cup"),
    (9, "Sunshine Airport", "Sunshine Airport", 3, "Star Cup"),
    (10, "Dolphin Shoals", "Dolfijnenparadijs", 3, "Star Cup"),
    (11, "Electrodrome", "Electrodome", 3, "Star Cup"),
    (12, "Mount Wario", "Wario's Winterberg", 3, "Star Cup"),
    (13, "Cloudtop Cruise", "Wildewolkenweg", 4, "Special Cup"),
    (14, "Bone Dry Dunes", "Dry Bowsers Woestijn", 4, "Special Cup"),
    (15, "Bowser's Castle", "Bowsers Kasteel", 4, "Special Cup"),
    (16, "Rainbow Road", "Regenboogbaan", 4, "Special Cup"),
    (17, "GCN Yoshi Circuit", "GCN Yoshi's Circuit", 5, "Egg Cup"),
    (18, "Excitebike Arena", "Excitebike-Arena", 5, "Egg Cup"),
    (19, "Dragon Driftway", "Drakendreef", 5, "Egg Cup"),
    (20, "Mute City", "Mute City", 5, "Egg Cup"),
    (21, "GCN Baby Park", "GCN Babypark", 6, "Crossing Cup"),
    (22, "GBA Cheese Land", "GBA Kaasland", 6, "Crossing Cup"),
    (23, "Wild Woods", "Wervelwoud", 6, "Crossing Cup"),
    (24, "Animal Crossing", "Animal Crossing", 6, "Crossing Cup"),
    (25, "Wii Moo Moo Meadows", "Wii Boe-Boe-Boerenland", 7, "Shell Cup"),
    (26, "GBA Mario Circuit", "GBA Mario's Circuit", 7, "Shell Cup"),
    (27, "DS Cheep Cheep Beach", "DS Cheep Cheep Strand", 7, "Shell Cup"),
    (28, "N64 Toad's Turnpike", "N64 Toads Tolweg", 7, "Shell Cup"),
    (29, "GCN Dry Dry Desert", "GCN Zinderende Zandvlakte", 8, "Banana Cup"),
    (30, "SNES Donut Plains 3", "SNES Donutvlakte 3", 8, "Banana Cup"),
    (31, "N64 Royal Raceway", "N64 Koninklijke Kartbaan", 8, "Banana Cup"),
    (32, "3DS DK Jungle", "3DS DK's Jungle", 8, "Banana Cup"),
    (33, "DS Wario Stadium", "DS Wario's Stadion", 9, "Leaf Cup"),
    (34, "GCN Sherbet Land", "GCN Ijzige Ijsbaan", 9, "Leaf Cup"),
    (35, "3DS Music Park", "3DS Muziekcircuit", 9, "Leaf Cup"),
    (36, "N64 Yoshi Valley", "N64 Yoshi's Vallei", 9, "Leaf Cup"),
    (37, "DS Tick-Tock Clock", "DS Tik-Tak-Klok", 10, "Lightning Cup"),
    (38, "3DS Piranha Plant Pipeway", "3DS Piranha Plant-Parkoers", 10, "Lightning Cup"),
    (39, "Wii Grumble Volcano", "Wii Dondervulkaan", 10, "Lightning Cup"),
    (40, "N64 Rainbow Road", "N64 Regenboogbaan", 10, "Lightning Cup"),
    (41, "Wii Wario's Gold Mine", "Wii Wario's Goudmijn", 11, "Triforce Cup"),
    (42, "SNES Rainbow Road", "SNES Regenboogbaan", 11, "Triforce Cup"),
    (43, "Ice Ice Outpost", "Toads Poolbasis", 11, "Triforce Cup"),
    (44, "Hyrule Circuit", "Hyrule-Circuit", 11, "Triforce Cup"),
    (45, "3DS Koopa City", "3DS Bowser City", 12, "Bell Cup"),
    (46, "GBA Ribbon Road", "GBA Sprintlint", 12, "Bell Cup"),
    (47, "Super Bell Subway", "Mario's Metro", 12, "Bell Cup"),
    (48, "Big Blue", "Big Blue", 12, "Bell Cup"),
    (49, "Tour Paris Promenade", "Tour Parijs-Promenade", 13, "Golden Dash Cup"),
    (50, "3DS Toad Circuit", "3DS Toads Circuit", 13, "Golden Dash Cup"),
    (51, "N64 Choco Mountain", "N64 Chocokloof", 13, "Golden Dash Cup"),
    (52, "Wii Coconut Mall", "Wii Kokosnootplaza", 13, "Golden Dash Cup"),
    (53, "Tour Tokyo Blur", "Tour Tokio-Toer", 14, "Lucky Cat Cup"),
    (54, "DS Shroom Ridge", "DS Paddenstoelenpas", 14, "Lucky Cat Cup"),
    (55, "GBA Sky Garden", "GBA Wolkenhof", 14, "Lucky Cat Cup"),
    (56, "Ninja Hideaway", "Ninjaschool", 14, "Lucky Cat Cup"),
    (57, "Tour New York Minute", "Tour New York Drive", 15, "Turnip Cup"),
    (58, "SNES Mario Circuit 3", "SNES Mario Circuit 3", 15, "Turnip Cup"),
    (59, "N64 Kalimari Desert", "N64 Wildwestwoestijn", 15, "Turnip Cup"),
    (60, "DS Waluigi Pinball", "DS Waluigi's Flipperkast", 15, "Turnip Cup"),
    (61, "Tour Sydney Sprint", "Tour Sydney-Sprint", 16, "Propeller Cup"),
    (62, "GBA Snow Land", "GBA Sneeuwland", 16, "Propeller Cup"),
    (63, "Wii Mushroom Gorge", "Wii Paddenstoelengrot", 16, "Propeller Cup"),
    (64, "Sky-High Sundae", "Stracciatellastraat", 16, "Propeller Cup"),
    (65, "Tour London Loop", "Tour Londen-Ronde", 17, "Rock Cup"),
    (66, "GBA Boo Lake", "GBA Boo-Bruggen", 17, "Rock Cup"),
    (67, "3DS Alpine Pass", "3DS Kampioensberg", 17, "Rock Cup"),
    (68, "Wii Maple Treeway", "Wii Wigglers Woud", 17, "Rock Cup"),
    (69, "Tour Berlin Byways", "Tour Berlijn-Bezoek", 18, "Moon Cup"),
    (70, "DS Peach Gardens", "DS Peach' Paleistuin", 18, "Moon Cup"),
    (71, "Merry Mountain", "Dennenboomdorp", 18, "Moon Cup"),
    (72, "3DS Rainbow Road", "3DS Regenboogbaan", 18, "Moon Cup"),
    (73, "Tour Amsterdam Drift", "Tour Afslag Amsterdam", 19, "Fruit Cup"),
    (74, "GBA Riverside Park", "GBA Rivieroever", 19, "Fruit Cup"),
    (75, "Wii DK's Snowboard Cross", "Wii DK's Skipark", 19, "Fruit Cup"),
    (76, "Yoshi's Island", "Yoshi's Eiland", 19, "Fruit Cup"),
    (77, "Tour Bangkok Rush", "Tour Bangkok-Break", 20, "Boomerang Cup"),
    (78, "DS Mario Circuit", "DS Mario's Circuit", 20, "Boomerang Cup"),
    (79, "GCN Waluigi Stadium", "GCN Waluigi's Stadion", 20, "Boomerang Cup"),
    (80, "Tour Singapore Speedway", "Tour Signapore-Skyline", 20, "Boomerang Cup"),
    (81, "Tour Athens Dash", "Tour Athene-Trip", 21, "Feather Cup"),
    (82, "GCN Daisy Cruiser", "GCN Daisy's Cruiseschip", 21, "Feather Cup"),
    (83, "Wii Moonview Highway", "Wii Maanlichtlaan", 21, "Feather Cup"),
    (84, "Squeaky Clean Sprint", "Badderbaan", 21, "Feather Cup"),
    (85, "Tour Los Angeles Laps", "Tour Los Angeles-Boulevard", 22, "Cherry Cup"),
    (86, "GBA Sunset Wilds", "GBA Schemersteppe", 22, "Cherry Cup"),
    (87, "Wii Koopa Cape", "Wii Kaap Koopa", 22, "Cherry Cup"),
    (88, "Tour Vancouver Velocity", "Tour Vancouver-Route", 22, "Cherry Cup"),
    (89, "Tour Rome Avanti", "Tour Rome-Run", 23, "Acorn Cup"),
    (90, "GCN DK Mountain", "GCN DK-Gebergte", 23, "Acorn Cup"),
    (91, "Wii Daisy Circuit", "Wii Daisy's Circuit", 23, "Acorn Cup"),
    (92, "Piranha Plant Cove", "Piranha Plant-Lagune", 23, "Acorn Cup"),
    (93, "Tour Madrid Drive", "Tour Madrid-Rit", 24, "Spiny Cup"),
    (94, "3DS Rosalina's Ice World", "3DS Rosalina's Ijsplaneet", 24, "Spiny Cup"),
    (95, "SNES Bowser Castle 3", "SNES Bowsers Kasteel 3", 24, "Spiny Cup"),
    (96, "Wii Rainbow Road", "Wii Regenboogbaan", 24, "Spiny Cup"),
]

def match_track_name(track_name: str, track_list: List[Tuple[int, str, str, int, str]]) -> str:
    """Match a track name to its English equivalent."""
    english_names = {track[1] for track in track_list}
    dutch_to_english = {track[2]: track[1] for track in track_list}

    if track_name in english_names:
        return track_name
    if track_name in dutch_to_english:
        return dutch_to_english[track_name]

    all_names = english_names.union(dutch_to_english.keys())
    best_match = difflib.get_close_matches(track_name, all_names, n=1, cutoff=0.85)

    if best_match:
        match = best_match[0]
        return dutch_to_english.get(match, match)

    return track_name

def get_cup_name(track_name: str, track_list: List[Tuple[int, str, str, int, str]]) -> str:
    """Get the Cup Name corresponding to a track."""
    for track in track_list:
        if track_name == track[1]:  # Match English name
            return track[4]
    return ""


def match_track_name(track_name: str, track_list: List[Tuple[int, str, str, int, str]]) -> str:
    """Match a track name to the closest name in the official track list."""
    # Combine English and Dutch names into a flat list
    all_names = [track[1] for track in track_list] + [track[2] for track in track_list]

    # Compute similarity scores for all names
    scores = [(difflib.SequenceMatcher(None, track_name, name).ratio(), name) for name in all_names]

    # Get the name with the highest similarity score
    best_match = max(scores, key=lambda x: x[0], default=(0, track_name))[1]

    return best_match

def preprocess_name(name):
    """Preprocess the name by removing special characters, normalizing case and whitespace."""
    name = re.sub(r'\W+', ' ', name)  # Remove non-alphanumeric characters
    name = name.strip().lower()  # Normalize case and strip leading/trailing whitespace
    name = re.sub(r'\s+', ' ', name)  # Normalize multiple spaces to single space
    return name

def soundex_similarity(name1, name2):
    """Calculate Soundex similarity as binary (0 or 1)."""
    return 1 if soundex(name1) == soundex(name2) else 0

def weighted_similarity(name1, name2):
    """Calculate a weighted similarity score based on difflib, string length, Jaro-Winkler, and Soundex."""
    # Preprocess names
    name1 = preprocess_name(name1)
    name2 = preprocess_name(name2)

    # Difflib similarity ratio
    difflib_score = difflib.SequenceMatcher(None, name1, name2).ratio()

    # String length difference (normalized)
    length_score = 1 - abs(len(name1) - len(name2)) / max(len(name1), len(name2), 1)

    # Jaro-Winkler similarity ratio
    jaro_winkler_score = textdistance.jaro_winkler(name1, name2)

    # Soundex similarity (binary)
    soundex_score = soundex_similarity(name1, name2)

    # Weights for each score
    weights = {
        'difflib': 0.3,
        'length': 0.4,
        'jaro_winkler': 0.3,
        'soundex': 0.0
    }

    # Calculate weighted score
    weighted_score = (weights['difflib'] * difflib_score +
                      weights['length'] * length_score +
                      weights['jaro_winkler'] * jaro_winkler_score +
                      weights['soundex'] * soundex_score)

    return weighted_score

def match_names(previous_names_with_indices, current_names):
    """Match names between a list of previous names with indices and current names based on exact match and weighted similarity."""
    matches = {}
    used_previous_names = set()
    used_current_names = set()
    exact_match_rows = set()

    previous_names = [name for name, _ in previous_names_with_indices]

    # Step 1: Exact Match
    for curr_name in current_names:
        if curr_name in previous_names:
            idx = previous_names.index(curr_name)
            row_idx = previous_names_with_indices[idx][1]
            matches[curr_name] = (curr_name, row_idx)
            used_previous_names.add(curr_name)
            used_current_names.add(curr_name)
            exact_match_rows.add(row_idx)

    # Step 2: Similarity Scores for remaining names, excluding exact match rows
    similarity_scores = []
    for curr_name in current_names:
        if curr_name not in used_current_names:
            for prev_name, prev_idx in previous_names_with_indices:
                if prev_name not in used_previous_names and prev_idx not in exact_match_rows:
                    score = weighted_similarity(prev_name, curr_name)
                    similarity_scores.append((score, prev_name, curr_name, prev_idx))

    # Sort similarity scores in descending order
    similarity_scores.sort(reverse=True, key=lambda x: x[0])

    # Match based on similarity scores
    for score, prev_name, curr_name, prev_idx in similarity_scores:
        if curr_name not in used_current_names and prev_name not in used_previous_names:
            matches[curr_name] = (prev_name, prev_idx)
            used_current_names.add(curr_name)
            used_previous_names.add(prev_name)

    # Handle remaining unmatched names by assigning them in order
    remaining_current_names = [name for name in current_names if name not in used_current_names]
    remaining_previous_names = [name for name, idx in previous_names_with_indices if name not in used_previous_names]

    for curr_name, prev_name in zip(remaining_current_names, remaining_previous_names):
        prev_idx = previous_names.index(prev_name)
        matches[curr_name] = (prev_name, prev_idx)

    # Add unmatched current names as-is if they have no match at all
    for curr_name in remaining_current_names[len(remaining_previous_names):]:
        matches[curr_name] = (curr_name, None)

    return matches

def build_name_links(df, output_folder):
    """Build name links across races for each RaceClass."""
    name_links = defaultdict(list)
    all_player_names_df = {}

    for race_class, group in df.groupby('RaceClass'):
        races = sorted(group['RaceIDNumber'].unique())
        num_races = len(races)

        max_players = group.groupby('RaceIDNumber').size().max()

        player_names_df = pd.DataFrame(index=range(max_players), columns=range(num_races))

        initial_race_id = races[0]
        initial_names = group[group['RaceIDNumber'] == initial_race_id]['PlayerName'].tolist()

        # Ensure the lengths match by padding or truncating
        player_names_df[0] = initial_names + [None] * (len(player_names_df) - len(initial_names))

        if num_races < 2:
            single_race_df = group[['PlayerName', 'RaceIDNumber']].copy()
            single_race_df.columns = ['PlayerName', 0]  # Rename columns to fit expected format
            single_race_df = single_race_df.set_index('PlayerName').T
            all_player_names_df[race_class] = single_race_df
            continue

        for col_idx, race_id in enumerate(races[1:], start=1):
            current_race_names = group[group['RaceIDNumber'] == race_id]['PlayerName'].tolist()

            # Collect all names from previous rounds along with their indices
            all_previous_names_with_indices = [(player_names_df.at[row_idx, prev_col], row_idx)
                                               for prev_col in range(col_idx)
                                               for row_idx in range(max_players)
                                               if pd.notna(player_names_df.at[row_idx, prev_col])]

            # Match names using the improved match_names function
            matches = match_names(all_previous_names_with_indices, current_race_names)

            used_rows_exact = set()  # Track rows used by exact matches
            used_rows_similarity = set()  # Track rows used by similarity matches

            # Assign exact matched names to the DataFrame using the row indices
            for curr_name in current_race_names:
                if curr_name in matches:
                    matched_name, row_idx = matches[curr_name]
                    if row_idx is not None and row_idx not in used_rows_exact:
                        player_names_df.at[row_idx, col_idx] = curr_name
                        used_rows_exact.add(row_idx)

            # Assign similarity matched names to remaining available rows
            for curr_name in current_race_names:
                if curr_name in matches and curr_name not in player_names_df.iloc[:, col_idx].dropna().values:
                    matched_name, row_idx = matches[curr_name]
                    if row_idx is not None and row_idx not in used_rows_exact and row_idx not in used_rows_similarity:
                        for i in range(max_players):
                            if i not in used_rows_exact and i not in used_rows_similarity and pd.isna(player_names_df.at[i, col_idx]):
                                player_names_df.at[i, col_idx] = curr_name
                                used_rows_similarity.add(i)
                                break

            # Fill remaining unmatched names
            used_names = set(player_names_df.iloc[:, col_idx].dropna().values)
            for row_idx, name in enumerate(current_race_names):
                if name not in used_names:
                    for i in range(max_players):
                        if i not in used_rows_exact and i not in used_rows_similarity and pd.isna(player_names_df.at[i, col_idx]):
                            player_names_df.at[i, col_idx] = name
                            used_names.add(name)
                            used_rows_similarity.add(i)
                            break

        for idx in range(max_players):
            name_link = player_names_df.loc[idx].dropna().tolist()
            if name_link:
                most_common_name = Counter(name_link).most_common(1)[0][0]
                for name in name_link:
                    name_links[name].append(most_common_name)

        output_path = os.path.join(output_folder, f'linking_{race_class}.xlsx')
        player_names_df.to_excel(output_path, index=False)

        all_player_names_df[race_class] = player_names_df

    return name_links, all_player_names_df



def standardize_names(player_names_df):
    """Determine the most frequent name in each row and create a mapping."""
    name_mapping = {}
    standardized_names = {}

    for idx in range(len(player_names_df)):
        name_link = player_names_df.loc[idx].dropna().tolist()
        if name_link:
            most_common_name = Counter(name_link).most_common(1)[0][0]
            for name in name_link:
                name_mapping[name] = (most_common_name, idx)  # Store the name and the row index
                standardized_names[idx] = most_common_name  # Store the standardized name by row index

    return name_mapping, standardized_names

def standardize_player_names(df, output_folder):
    """Standardize player names within each RaceClass."""
    standardized_names = pd.DataFrame()
    name_mapping = {}
    standardized_names_dict = {}
    name_links, all_player_names_df = build_name_links(df, output_folder)

    for race_class, player_names_df in all_player_names_df.items():

        # Handle cases where there is only one race
        if player_names_df.shape[1] < 2:
            continue

        local_name_mapping, local_standardized_names_dict = standardize_names(player_names_df)
        name_mapping.update(local_name_mapping)
        standardized_names_dict.update(local_standardized_names_dict)
        group = df[df['RaceClass'] == race_class].copy()

        def get_standardized_name(row):
            player_name = row['PlayerName']
            if player_name in local_name_mapping:
                return standardized_names_dict[local_name_mapping[player_name][1]]
            else:
                return player_name  # Fallback to original name if not found

        group.loc[:, 'FixPlayerName'] = group.apply(get_standardized_name, axis=1)

        standardized_names = pd.concat([standardized_names, group], ignore_index=True)


    return standardized_names


def process_images_in_folder(folder_path: str) -> None:
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])

    # Check if the folder is empty
    if not image_files:
        print("Error: The Frames folder within the Output_Results folder is empty or contains no PNG files. Please Run Step 2 first.")
        sys.exit(1)  # Exit the script with a non-zero status

    results = []

    script_dir = os.path.dirname(__file__)  # Directory of the script
    text_detected_folder = os.path.join(script_dir, 'Output_Results', 'Debug', 'Score_Frames')
    if not os.path.exists(text_detected_folder):
        os.makedirs(text_detected_folder)

    linking_data_folder = os.path.join(script_dir, 'Output_Results', 'Debug')
    if not os.path.exists(linking_data_folder):
        os.makedirs(linking_data_folder)

    grouped_images = {}
    for image_file in image_files:
        parts = image_file.split('+')
        if len(parts) < 3:
            logging.warning(f"Skipping file with unexpected format: {image_file}")
            continue

        race_class = parts[0]
        try:
            race_id_number = int(parts[1][-3:])
        except ValueError:
            logging.warning(f"Skipping file with invalid race ID number: {image_file}")
            continue

        frame_content = parts[2].replace('.png', '')
        if frame_content == "1RaceNumber" or frame_content == "3TotalScore":
            continue

        key = (race_class, race_id_number)
        if key not in grouped_images:
            grouped_images[key] = []
        grouped_images[key].append((frame_content, os.path.join(folder_path, image_file)))

    for (race_class, race_id_number), images in grouped_images.items():
        if len(images) != 2:
            continue

        # Adjusted logging statement to show only the base name of the images
        base_names = [os.path.basename(image_path) for _, image_path in images]
        logging.info(f"Processing: {base_names}")

        track_name_image = None
        race_score_images = []

        for frame_content, image_path in images:
            if frame_content == "0TrackName":
                track_name_image = image_path
            elif frame_content == "2RaceScore":
                race_score_images.append(image_path)

        if not track_name_image or len(race_score_images) != 1:
            continue

        track_name_img = cv2.imread(track_name_image)
        coordinates = {"TrackName": [((319, 633), (925, 685))]}
        # Extract track name data with confidence
        track_name_data, _ = extract_text_with_confidence(track_name_image, coordinates, 'eng', '--psm 7')

        # Combine the extracted text into a single string
        raw_track_name_text = " ".join(track_name_data['TrackName']).strip()

        # Match the extracted track name to the closest name in the official track list
        track_name_text = match_track_name(raw_track_name_text, tracks_list)

        for race_score_image in race_score_images:
            processed_img = process_image(race_score_image)
            processed_img_pil = Image.fromarray(processed_img).convert('RGB')
            scale_factor = 5
            scaled_image = processed_img_pil.resize(
                (processed_img_pil.width * scale_factor, processed_img_pil.height * scale_factor), Image.NEAREST)

            start_coords_run1 = [(830, 71), (843, 71)]
            start_coords_run1 = scale_coords(start_coords_run1, scale_factor)

            red_pixels_run1 = {
                "top_middle": (7, 2), "left_middle": (2, 5), "middle_middle": (7, 5),
                "right_middle": (11, 5), "left_bottom": (2, 13), "middle_bottom": (7, 13),
                "right_bottom": (11, 13), "middle_bottom_edge": (7, 17), "center": (7, 9)
            }
            red_pixels_run1 = scale_pixel_positions(red_pixels_run1, scale_factor)

            row_offset_run1 = 52 * scale_factor
            box_dims_run1 = (13 * scale_factor, 19 * scale_factor)
            num_rows_run1 = 12
            boxes_per_row_run1 = 2

            race_points = detect_digits_in_image(
                scaled_image, start_coords_run1, row_offset_run1, box_dims_run1,
                red_pixels_run1, num_rows_run1, boxes_per_row_run1
            )

            start_coords_run2 = [(916, 66), (933, 66), (950, 66)]
            start_coords_run2 = scale_coords(start_coords_run2, scale_factor)

            red_pixels_run2 = {
                "top_middle": (8, 2), "left_middle": (2, 7), "middle_middle": (8, 7),
                "right_middle": (13, 7), "left_bottom": (2, 16), "middle_bottom": (8, 16),
                "right_bottom": (13, 16), "middle_bottom_edge": (8, 21), "center": (8, 11)
            }
            red_pixels_run2 = scale_pixel_positions(red_pixels_run2, scale_factor)

            row_offset_run2 = 52 * scale_factor
            box_dims_run2 = (16 * scale_factor, 24 * scale_factor)
            num_rows_run2 = 12
            boxes_per_row_run2 = 3

            old_total_score = detect_digits_in_image(
                scaled_image, start_coords_run2, row_offset_run2, box_dims_run2,
                red_pixels_run2, num_rows_run2, boxes_per_row_run2
            )

            annotated_image_path = os.path.join(text_detected_folder, f'annotated_{os.path.basename(race_score_image)}')
            scaled_image_resized = scaled_image.resize((processed_img_pil.width, processed_img_pil.height),
                                                       Image.NEAREST)
            scaled_image_resized.save(annotated_image_path)

            player_name_text, confidence_scores = extract_text_with_confidence(
                annotated_image_path, {"player_name": [
                    ((428, 52), (620, 96)), ((428, 104), (620, 148)),
                    ((428, 156), (620, 200)), ((428, 208), (620, 252)),
                    ((428, 260), (620, 304)), ((428, 312), (620, 356)),
                    ((428, 364), (620, 408)), ((428, 416), (620, 460)),
                    ((428, 468), (620, 512)), ((428, 520), (620, 564)),
                    ((428, 572), (620, 617)), ((428, 624), (620, 669))
                ]},
                lang='eng',
                config='--psm 7'
            )

            # Initialize filtered results and temporary storage
            filtered_player_names = []
            filtered_confidences = []

            # Filter invalid rows
            for i in range(len(player_name_text["player_name"])):
                player_name = player_name_text["player_name"][i]
                confidence_score = confidence_scores[i] if i < len(confidence_scores) else "N/A"

                # Skip invalid rows
                if confidence_score == "N/A":
                    #print(
                    #    f"Skipping entry with unavailable confidence: Player Name: {player_name}, Confidence: {confidence_score}")
                    continue
                if (confidence_score < 50 and len(player_name) <= 2) or (
                        len(player_name) <= 3 and not any(char.isalpha() for char in player_name)
                ):
                    #print(f"Skipping invalid entry: Player Name: {player_name}, Confidence: {confidence_score}")
                    continue

                # Append valid entries to filtered lists
                filtered_player_names.append(player_name)
                filtered_confidences.append(confidence_score)

            # Calculate the number of players after filtering
            num_players = len(filtered_player_names)

            # Process filtered players
            for i, player_name in enumerate(filtered_player_names):
                race_position = i + 1
                race_points_fix = get_race_points(race_position, num_players)  # Dynamic race points

                # Calculate scores
                old_total_score_value = 0  # Replace with actual logic for old total score
                new_total_score = old_total_score_value + race_points_fix

                # Build the results list
                results.append([
                    race_class, race_id_number, track_name_text, race_position, player_name,
                    race_points_fix
                ])

            # Print confidence scores for debugging
            #for player_name, confidence_score in zip(filtered_player_names, filtered_confidences):
            #    print(f"Player Name: {player_name}, Confidence Score: {confidence_score}")

    # Create DataFrame without ConfidenceScore
    df = pd.DataFrame(results, columns=[
        "RaceClass", "RaceIDNumber", "TrackName", "RacePosition", "PlayerName",
        "RacePoints"
    ])

    # Add the CupName column
    df['CupName'] = df['TrackName'].apply(lambda name: get_cup_name(name, tracks_list))

    # Add the TrackID column
    df['TrackID'] = df['TrackName'].apply(lambda name: next((track[0] for track in tracks_list if track[1] == name), None))

    # Add standardized player names
    df = standardize_player_names(df, linking_data_folder)

    # Add the FixOldTotalScore and FixNewTotalScore columns
    df['OldTotalScore'], df['NewTotalScore'] = 0, 0
    for i, row in df.iterrows():
        if row['RaceIDNumber'] > 1:
            prev_row = df[(df['RaceClass'] == row['RaceClass']) &
                          (df['RaceIDNumber'] == row['RaceIDNumber'] - 1) &
                          (df['FixPlayerName'] == row['FixPlayerName'])]
            if not prev_row.empty:
                prev_score = prev_row['NewTotalScore'].values[0]
                df.at[i, 'OldTotalScore'] = prev_score
        df.at[i, 'NewTotalScore'] = df.at[i, 'RacePoints'] + df.at[i, 'OldTotalScore']

    # Reorder columns for the final output
    desired_order = [
        "RaceClass", "RaceIDNumber", "TrackName", "TrackID", "CupName", "RacePosition", "PlayerName",
        "FixPlayerName", "RacePoints", "OldTotalScore", "NewTotalScore"

    ]
    df = df[desired_order]

    # Save to Excel
    output_excel_path = os.path.join(folder_path, '..', "Tournament_Results.xlsx")
    df.to_excel(output_excel_path, index=False)
    logging.info(f'Results saved to {output_excel_path}')


# Folder path to the images
script_dir = os.path.dirname(__file__)  # Directory of the script
folder_path = os.path.join(script_dir, 'Output_Results', 'Frames')
process_images_in_folder(folder_path)

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_run_time

# Convert elapsed time to minutes and seconds
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

# Print the elapsed time in mm:ss format
print(f"Runtime was: {minutes:02}:{seconds:02}")