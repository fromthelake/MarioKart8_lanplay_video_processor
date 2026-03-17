from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .project_paths import PROJECT_ROOT

DEFAULT_SCORE_LAYOUT_ID = "lan2_split_2p"
LAN1_SCORE_LAYOUT_ID = "lan1_full_1p"
SCORE_LAYOUT_SHIFT_X = 250

PLAYER_NAME_COORDS_LAN2 = [
    ((428, 52), (620, 96)), ((428, 104), (620, 148)),
    ((428, 156), (620, 200)), ((428, 208), (620, 252)),
    ((428, 260), (620, 304)), ((428, 312), (620, 356)),
    ((428, 364), (620, 408)), ((428, 416), (620, 460)),
    ((428, 468), (620, 512)), ((428, 520), (620, 564)),
    ((428, 572), (620, 617)), ((428, 624), (620, 669)),
]
RACE_POINTS_COORDS_LAN2 = [
    ((825, 52), (861, 96)), ((825, 104), (861, 148)),
    ((825, 156), (861, 200)), ((825, 208), (861, 252)),
    ((825, 260), (861, 304)), ((825, 312), (861, 356)),
    ((825, 364), (861, 408)), ((825, 416), (861, 460)),
    ((825, 468), (861, 512)), ((825, 520), (861, 564)),
    ((825, 572), (861, 617)), ((825, 624), (861, 669)),
]
TOTAL_POINTS_COORDS_LAN2 = [
    ((910, 52), (973, 96)), ((910, 104), (973, 148)),
    ((910, 156), (973, 200)), ((910, 208), (973, 252)),
    ((910, 260), (973, 304)), ((910, 312), (973, 356)),
    ((910, 364), (973, 408)), ((910, 416), (973, 460)),
    ((910, 468), (973, 512)), ((910, 520), (973, 564)),
    ((910, 572), (973, 617)), ((910, 624), (973, 669)),
]


def _shift_box(box: Tuple[int, int, int, int], shift_x: int) -> Tuple[int, int, int, int]:
    x, y, width, height = box
    return (x + shift_x, y, width, height)


def _shift_rectangles(
    rectangles: List[Tuple[Tuple[int, int], Tuple[int, int]]], shift_x: int
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    return [
        ((x1 + shift_x, y1), (x2 + shift_x, y2))
        for (x1, y1), (x2, y2) in rectangles
    ]


@dataclass(frozen=True)
class ScoreLayout:
    layout_id: str
    score_anchor_roi: Tuple[int, int, int, int]
    scoreboard_points_roi: Tuple[int, int, int, int]
    twelfth_place_check_roi: Tuple[int, int, int, int]
    player_name_coords: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    base_position_strip_roi: Tuple[Tuple[int, int], Tuple[int, int]]
    character_roi_left: int
    race_points_coords: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    total_points_coords: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    race_digit_starts: List[Tuple[int, int]]
    total_digit_starts: List[Tuple[int, int]]


_SCORE_LAYOUTS: Dict[str, ScoreLayout] = {
    DEFAULT_SCORE_LAYOUT_ID: ScoreLayout(
        layout_id=DEFAULT_SCORE_LAYOUT_ID,
        score_anchor_roi=(315, 57, 52, 610),
        scoreboard_points_roi=(290, 32, 102, 660),
        twelfth_place_check_roi=(313, 632, 651, 88),
        player_name_coords=PLAYER_NAME_COORDS_LAN2,
        base_position_strip_roi=((315, 57), (367, 667)),
        character_roi_left=377,
        race_points_coords=RACE_POINTS_COORDS_LAN2,
        total_points_coords=TOTAL_POINTS_COORDS_LAN2,
        race_digit_starts=[(830, 71), (843, 71)],
        total_digit_starts=[(916, 66), (933, 66), (950, 66)],
    ),
    LAN1_SCORE_LAYOUT_ID: ScoreLayout(
        layout_id=LAN1_SCORE_LAYOUT_ID,
        score_anchor_roi=_shift_box((315, 57, 52, 610), SCORE_LAYOUT_SHIFT_X),
        scoreboard_points_roi=_shift_box((290, 32, 102, 660), SCORE_LAYOUT_SHIFT_X),
        twelfth_place_check_roi=_shift_box((313, 632, 651, 88), SCORE_LAYOUT_SHIFT_X),
        player_name_coords=_shift_rectangles(PLAYER_NAME_COORDS_LAN2, SCORE_LAYOUT_SHIFT_X),
        base_position_strip_roi=((565, 57), (617, 667)),
        character_roi_left=377 + SCORE_LAYOUT_SHIFT_X,
        race_points_coords=_shift_rectangles(RACE_POINTS_COORDS_LAN2, SCORE_LAYOUT_SHIFT_X),
        total_points_coords=_shift_rectangles(TOTAL_POINTS_COORDS_LAN2, SCORE_LAYOUT_SHIFT_X),
        race_digit_starts=[(1080, 71), (1093, 71)],
        total_digit_starts=[(1166, 66), (1183, 66), (1200, 66)],
    ),
}


def get_score_layout(layout_id: str | None) -> ScoreLayout:
    return _SCORE_LAYOUTS.get(str(layout_id or "").strip(), _SCORE_LAYOUTS[DEFAULT_SCORE_LAYOUT_ID])


def all_score_layouts() -> List[ScoreLayout]:
    return [_SCORE_LAYOUTS[DEFAULT_SCORE_LAYOUT_ID], _SCORE_LAYOUTS[LAN1_SCORE_LAYOUT_ID]]


def score_layout_id_from_filename(image_path: str | os.PathLike[str] | None) -> str:
    if not image_path:
        return DEFAULT_SCORE_LAYOUT_ID
    stem_parts = Path(str(image_path)).stem.split("+")
    for part in reversed(stem_parts):
        if part in _SCORE_LAYOUTS:
            return part
    return DEFAULT_SCORE_LAYOUT_ID


def build_score_frame_filename(video_label: str, race_number: int, frame_content: str, score_layout_id: str) -> str:
    return f"{video_label}+Race_{race_number:03}+{frame_content}+{score_layout_id}.png"


def draw_score_layout_demo(
    image: np.ndarray,
    layout_id: str,
    frame_content: str,
    output_path: str | os.PathLike[str],
) -> None:
    layout = get_score_layout(layout_id)
    annotated = image.copy()
    colors = {
        "score_anchor": (0, 255, 255),
        "score_points": (0, 165, 255),
        "twelfth_check": (0, 0, 255),
        "position_strip": (255, 255, 0),
        "character": (0, 255, 0),
        "player_name": (255, 0, 255),
        "race_points": (255, 128, 0),
        "total_points": (255, 0, 0),
    }

    def _draw_box(name: str, roi: Tuple[int, int, int, int], color_key: str) -> None:
        x, y, width, height = roi
        cv2.rectangle(annotated, (x, y), (x + width, y + height), colors[color_key], 2)
        cv2.putText(
            annotated,
            name,
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colors[color_key],
            1,
            cv2.LINE_AA,
        )

    def _draw_rectangles(name: str, rectangles: List[Tuple[Tuple[int, int], Tuple[int, int]]], color_key: str) -> None:
        for index, ((x1, y1), (x2, y2)) in enumerate(rectangles, start=1):
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colors[color_key], 1)
            if index == 1:
                cv2.putText(
                    annotated,
                    name,
                    (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    colors[color_key],
                    1,
                    cv2.LINE_AA,
                )

    _draw_box("score anchor", layout.score_anchor_roi, "score_anchor")
    _draw_box("scoreboard points", layout.scoreboard_points_roi, "score_points")
    _draw_box("12th check", layout.twelfth_place_check_roi, "twelfth_check")
    (px1, py1), (px2, py2) = layout.base_position_strip_roi
    cv2.rectangle(annotated, (px1, py1), (px2, py2), colors["position_strip"], 2)
    cv2.putText(annotated, "position strip", (px1, max(18, py1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors["position_strip"], 1, cv2.LINE_AA)
    _draw_rectangles("player names", layout.player_name_coords, "player_name")
    _draw_rectangles("race points", layout.race_points_coords, "race_points")
    _draw_rectangles("total points", layout.total_points_coords, "total_points")
    first_character = ((layout.character_roi_left, 47), (layout.character_roi_left + 48, 47 + 626))
    cv2.rectangle(annotated, first_character[0], first_character[1], colors["character"], 2)
    cv2.putText(
        annotated,
        "character column",
        (first_character[0][0], max(18, first_character[0][1] - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        colors["character"],
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"{frame_content} | {layout.layout_id}",
        (18, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_file), annotated)


def score_demo_output_path(video_label: str, race_number: int, frame_content: str, score_layout_id: str) -> Path:
    return PROJECT_ROOT / "Output_Results" / "Debug" / "Score_Layout_Demos" / (
        f"{video_label}+Race_{race_number:03}+{frame_content}+{score_layout_id}+roi_demo.png"
    )
