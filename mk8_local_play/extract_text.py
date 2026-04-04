import os
import argparse
import csv
import cv2
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import logging
import time
import difflib
import openpyxl
import re
import threading
import warnings
from collections import defaultdict
from functools import lru_cache
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

try:
    import easyocr
except Exception:
    easyocr = None

try:
    import torch
except Exception:
    torch = None

warnings.filterwarnings(
    "ignore",
    message=r"'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used\.",
    category=UserWarning,
)
from .app_runtime import easyocr_gpu_enabled as runtime_easyocr_gpu_enabled, load_app_config
from .extract_common import (
    build_video_identity,
    debug_score_frame_path,
    debug_score_frame_variant_path,
    extract_exported_frame_number,
    find_anchor_frame_path,
    find_score_bundle_race_context_paths,
    find_score_bundle_anchor_path,
    iter_video_race_dirs,
)
from .console_logging import LOGGER
from .ocr_export import build_completion_payload, build_player_count_summary_lines
from .ocr_name_matching import (
    append_identity_ambiguity_review_notes,
    append_identity_relink_review_notes,
    choose_canonical_name,
    compact_identity_labels,
    merge_fragmented_identity_aliases,
    preprocess_name,
    reconcile_connection_reset_identities,
    resolve_duplicate_name_identity_chains,
    standardize_player_names,
    weighted_similarity,
)
from .ocr_common import find_metadata_entry, load_consensus_frame_entries, load_exported_frame_metadata
from .extract_common import write_export_image
from .low_res_identity import apply_low_res_identity_pipeline, is_low_res_height
from .name_unicode import (
    allowed_name_char_ratio,
    distinct_visible_name_count,
    unknown_name_chars,
    visible_name_characters,
    visible_name_length,
)
from .ocr_scoreboard_consensus import (
    TOTAL_SCORE_CONSENSUS_WINDOW_SIZE,
    best_character_matches,
    build_consensus_observation,
    build_race_warning_messages,
    call_matrix_summary_lines,
    character_row_roi,
    character_shortlist_summary_lines,
    exact_total_score_fallback,
    load_character_templates,
    observation_stage_summary_lines,
    parse_detected_int,
    record_call_matrix,
    reset_character_shortlist_state,
    reset_call_matrix_stats,
    reset_observation_stage_stats,
)
from .ocr_session_validation import apply_session_validation
from .ocr_scoring_policy import apply_temporary_player_drop_scoring_policy
from .project_paths import PROJECT_ROOT
from .score_layouts import get_score_layout, score_layout_id_from_filename
from .track_metadata import load_track_tuples
from .game_catalog import load_game_catalog

POSITION_TEMPLATE_COEFF_COLUMNS = [f"PositionTemplate{template_index:02}_Coeff" for template_index in range(1, 13)]
PLACEHOLDER_NAME_PREFIX = "PlayerNameMissing_"
PLACEHOLDER_RESCUE_MIN_SUPPORT = 3
PLACEHOLDER_RESCUE_MIN_ROW_SCORE = 140.0
PLACEHOLDER_RESCUE_MIN_MARGIN = 35.0
PLACEHOLDER_FORCED_CHOICE_MIN_SUPPORT = 2
PLACEHOLDER_FORCED_CHOICE_MIN_ROW_SCORE = 160.0
PLACEHOLDER_FORCED_SINGLE_HIT_MIN_ROW_SCORE = 190.0
BLACK_BLUE_VARIANT_FAMILIES = {
    "Yoshi": ("Black Yoshi", "Blue Yoshi"),
    "Shy Guy": ("Black Shy Guy", "Blue Shy Guy"),
    "Birdo": ("Black Birdo", "Blue Birdo"),
}
BLACK_BLUE_VARIANT_NAMES = {
    name
    for pair in BLACK_BLUE_VARIANT_FAMILIES.values()
    for name in pair
}
BLACK_BLUE_VARIANT_MARGIN = 0.015
BLACK_BLUE_VARIANT_MIN_SCORE = 0.52
CHARACTER_VARIANT_FAMILY_ROSTER_NAMES = {
    7: "Birdo",
    8: "Yoshi",
    11: "Shy Guy",
}
CHARACTER_VARIANT_FAMILY_ROSTER_INDICES = set(CHARACTER_VARIANT_FAMILY_ROSTER_NAMES.keys())
EXPLICIT_CHARACTER_VARIANT_FAMILIES = {
    "Peach": ("Peach", "Cat Peach", "Baby Peach", "Pink Gold Peach", "Peachette"),
}
VARIANT_FAMILY_REFINEMENT_MIN_SUPPORT = 3
VARIANT_FAMILY_REFINEMENT_MIN_DOMINANT_RATIO = 0.9
VARIANT_FAMILY_REFINEMENT_MIN_AVG_MARGIN = 0.5
VARIANT_FAMILY_REFINEMENT_METHOD = "variant_family_diagnostic_refine"

# Record the start time
start_run_time = time.time()
APP_CONFIG = load_app_config()
OCR_CONSENSUS_FRAMES = APP_CONFIG.ocr_consensus_frames
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
PLAYER_NAME_BATCH_FALLBACK_CONFIDENCE = max(
    0,
    min(100, int(os.environ.get("MK8_PLAYER_NAME_BATCH_FALLBACK_CONFIDENCE", "50"))),
)
PLAYER_NAME_BATCH_SEPARATOR_HEIGHT = max(
    0,
    int(os.environ.get("MK8_PLAYER_NAME_BATCH_SEPARATOR_HEIGHT", "10")),
)
PLAYER_NAME_BATCH_HORIZONTAL_PADDING = max(
    0,
    int(os.environ.get("MK8_PLAYER_NAME_BATCH_HORIZONTAL_PADDING", "0")),
)
PLAYER_NAME_BATCH_VERTICAL_PADDING = max(
    0,
    int(os.environ.get("MK8_PLAYER_NAME_BATCH_VERTICAL_PADDING", "0")),
)
PLAYER_NAME_BATCH_CONFIG = os.environ.get("MK8_PLAYER_NAME_BATCH_CONFIG", "--psm 6")
PLAYER_NAME_ROW_FALLBACK_ENABLED = os.environ.get("MK8_PLAYER_NAME_ROW_FALLBACK_ENABLED", "0").lower() not in {"0", "false", "no"}
RACE_SCORE_NAME_ROW_FALLBACK_ENABLED = os.environ.get("MK8_RACE_SCORE_NAME_ROW_FALLBACK_ENABLED", "1").lower() not in {"0", "false", "no"}
TOTAL_SCORE_NAME_ROW_FALLBACK_ENABLED = os.environ.get("MK8_TOTAL_SCORE_NAME_ROW_FALLBACK_ENABLED", "0").lower() not in {"0", "false", "no"}


@lru_cache(maxsize=1)
def _character_variant_family_lookup() -> dict[str, tuple[int, str]]:
    lookup: dict[str, tuple[int, str]] = {}
    for family_name, variant_names in EXPLICIT_CHARACTER_VARIANT_FAMILIES.items():
        for variant_name in variant_names:
            lookup[str(variant_name).strip()] = (-1, str(family_name))
    catalog = load_game_catalog()
    for character in catalog.characters:
        roster_index = int(character.roster_index)
        if roster_index not in CHARACTER_VARIANT_FAMILY_ROSTER_INDICES:
            continue
        lookup[str(character.name_uk).strip()] = (
            roster_index,
            CHARACTER_VARIANT_FAMILY_ROSTER_NAMES[roster_index],
        )
    return lookup


def resolve_character_variant_family_name(character_name: str | None) -> str:
    normalized = str(character_name or "").strip()
    if not normalized:
        return ""
    return _character_variant_family_lookup().get(normalized, (None, ""))[1]


def character_variant_family_templates(
    templates: List[Dict[str, object]],
    character_name: str | None,
) -> List[Dict[str, object]]:
    normalized = str(character_name or "").strip()
    if not normalized:
        return []

    family_lookup = _character_variant_family_lookup()
    family_entry = family_lookup.get(normalized)
    if family_entry is None:
        return []

    family_roster_index, _family_name = family_entry
    if int(family_roster_index) < 0:
        explicit_family_names = {
            str(name).strip()
            for name in EXPLICIT_CHARACTER_VARIANT_FAMILIES.get(str(_family_name), ())
        }
        return [
            template
            for template in templates
            if str(template.get("character_name", "")).strip() in explicit_family_names
        ]

    catalog_lookup = {
        str(character.name_uk).strip(): int(character.roster_index)
        for character in load_game_catalog().characters
    }
    family_templates = [
        template
        for template in templates
        if catalog_lookup.get(str(template.get("character_name", "")).strip()) == family_roster_index
    ]
    return family_templates


def build_character_variant_family_diagnostic_mask(
    family_templates: List[Dict[str, object]],
    family_name: str | None = None,
) -> np.ndarray:
    if not family_templates:
        return np.zeros((0, 0), dtype=bool)

    template_images = []
    alpha_masks = []
    for template in family_templates:
        template_images.append(np.asarray(template["template_image"], dtype=np.uint8))
        alpha_masks.append(np.asarray(template["template_alpha"], dtype=np.uint8))

    hsv_stack = np.stack(
        [cv2.cvtColor(template_image, cv2.COLOR_BGR2HSV).astype(np.float32) for template_image in template_images],
        axis=0,
    )
    alpha_stack = np.stack(alpha_masks, axis=0)
    alpha_presence = alpha_stack > 16
    alpha_mean = np.mean(alpha_presence, axis=0)
    alpha_mask = alpha_mean > 0.9
    alpha_var = np.var(alpha_presence.astype(np.float32), axis=0)
    hue_var = np.var(hsv_stack[:, :, :, 0], axis=0)
    sat_var = np.var(hsv_stack[:, :, :, 1], axis=0)
    sat_mean = np.mean(hsv_stack[:, :, :, 1], axis=0)
    val_mean = np.mean(hsv_stack[:, :, :, 2], axis=0)
    diagnostic_mask = alpha_mask & ((sat_var > 2000.0) | ((hue_var > 300.0) & (sat_mean > 40.0))) & (val_mean > 40.0)
    if str(family_name or "").strip() == "Peach":
        # Peach-family confusion is often driven by silhouette features such as cat ears that do
        # not survive the old "visible in almost every template" mask. Include pixels whose alpha
        # occupancy differs strongly across the family so Cat Peach can win on those shape cues.
        diagnostic_mask |= (
            (alpha_mean > 0.15)
            & (alpha_mean < 0.85)
            & (alpha_var > 0.12)
            & (val_mean > 25.0)
        )
    return diagnostic_mask


def diagnostic_character_variant_score(
    source_image: np.ndarray,
    template_image: np.ndarray,
    diagnostic_mask: np.ndarray,
) -> float:
    if source_image.size == 0 or template_image.size == 0 or diagnostic_mask.size == 0:
        return 0.0

    template_height, template_width = template_image.shape[:2]
    if source_image.shape[0] != template_height or source_image.shape[1] != template_width:
        source_image = cv2.resize(source_image, (template_width, template_height), interpolation=cv2.INTER_LINEAR)

    source_hsv = cv2.cvtColor(source_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    template_hsv = cv2.cvtColor(template_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    source_h = source_hsv[:, :, 0][diagnostic_mask]
    template_h = template_hsv[:, :, 0][diagnostic_mask]
    source_s = source_hsv[:, :, 1][diagnostic_mask]
    template_s = template_hsv[:, :, 1][diagnostic_mask]
    source_v = source_hsv[:, :, 2][diagnostic_mask]
    template_v = template_hsv[:, :, 2][diagnostic_mask]

    if source_h.size == 0:
        return 0.0

    hue_delta = np.abs(source_h - template_h)
    hue_delta = np.minimum(hue_delta, 180.0 - hue_delta)
    sat_diff = np.abs(source_s - template_s)
    val_diff = np.abs(source_v - template_v)

    score = 1.0
    score -= 0.55 * float(np.mean(hue_delta) / 90.0)
    score -= 0.30 * float(np.mean(sat_diff) / 255.0)
    score -= 0.15 * float(np.mean(val_diff) / 255.0)
    return max(0.0, min(1.0, score))


def refine_character_variant_families(df: pd.DataFrame, frames_folder: str | Path) -> pd.DataFrame:
    """Resolve stable color-variant families using family-specific diagnostic pixels."""
    if df.empty or "Character" not in df.columns or "FixPlayerName" not in df.columns:
        return df

    df = df.copy()
    for column_name in (
        "CharacterFamilyName",
        "CharacterFamilyBest",
        "CharacterFamilyBestIndex",
        "CharacterFamilyBestCoeff",
        "CharacterFamilySecond",
        "CharacterFamilySecondCoeff",
        "CharacterFamilyMargin",
    ):
        if column_name not in df.columns:
            df[column_name] = ""
    templates = load_character_templates()
    if not templates:
        return df

    template_by_name = {str(template["character_name"]): template for template in templates}
    family_template_cache: dict[str, tuple[list[dict[str, object]], np.ndarray]] = {}
    frame_cache: dict[tuple[str, int], tuple[np.ndarray | None, str]] = {}
    row_refinements: dict[int, dict[str, object]] = {}

    for row_index, row in df.iterrows():
        current_character = str(row.get("Character") or "").strip()
        family_name = resolve_character_variant_family_name(current_character)
        if not family_name:
            continue

        family_templates, diagnostic_mask = family_template_cache.get(family_name, (None, None))
        if family_templates is None:
            family_templates = character_variant_family_templates(templates, current_character)
            diagnostic_mask = build_character_variant_family_diagnostic_mask(family_templates, family_name=family_name)
            family_template_cache[family_name] = (family_templates, diagnostic_mask)
        if not family_templates or diagnostic_mask.size == 0 or int(np.count_nonzero(diagnostic_mask)) <= 0:
            continue

        race_class = str(row.get("RaceClass", "") or "")
        race_id = int(row.get("RaceIDNumber", 0) or 0)
        position = int(row.get("RacePosition", 0) or 0)
        if not race_class or race_id <= 0 or position <= 0:
            continue

        cache_key = (race_class, race_id)
        if cache_key not in frame_cache:
            preferred_frame = find_score_bundle_anchor_path(race_class, race_id, "2RaceScore")
            if preferred_frame is None:
                frame_cache[cache_key] = (None, "")
            else:
                frame_cache[cache_key] = (
                    cv2.imread(str(preferred_frame), cv2.IMREAD_COLOR),
                    score_layout_id_from_filename(preferred_frame),
                )
        frame_image, score_layout_id = frame_cache[cache_key]
        if frame_image is None:
            continue

        (x1, y1), (x2, y2) = character_row_roi(position - 1, score_layout_id=score_layout_id)
        row_roi = frame_image[y1:y2, x1:x2]
        if row_roi.size == 0:
            continue

        ranked_scores = []
        for family_template in family_templates:
            score = diagnostic_character_variant_score(
                row_roi,
                family_template["template_image"],
                diagnostic_mask,
            )
            ranked_scores.append(
                (
                    str(family_template["character_name"]),
                    int(family_template["character_index"]),
                    float(score) * 100.0,
                )
            )
        ranked_scores.sort(key=lambda item: item[2], reverse=True)
        if not ranked_scores:
            continue
        best_name, best_index, best_score = ranked_scores[0]
        second_name = ranked_scores[1][0] if len(ranked_scores) > 1 else ""
        second_score = ranked_scores[1][2] if len(ranked_scores) > 1 else 0.0
        row_refinements[row_index] = {
            "family_name": family_name,
            "winner_name": best_name,
            "winner_index": best_index,
            "winner_score": round(best_score, 1),
            "runner_up_name": second_name,
            "runner_up_score": round(second_score, 1),
            "winner_margin": round(best_score - second_score, 1),
        }

    if not row_refinements:
        return df

    for row_index, refinement in row_refinements.items():
        df.at[row_index, "CharacterFamilyName"] = str(refinement["family_name"])
        df.at[row_index, "CharacterFamilyBest"] = str(refinement["winner_name"])
        df.at[row_index, "CharacterFamilyBestIndex"] = int(refinement["winner_index"])
        df.at[row_index, "CharacterFamilyBestCoeff"] = round(float(refinement["winner_score"]), 1)
        df.at[row_index, "CharacterFamilySecond"] = str(refinement["runner_up_name"])
        df.at[row_index, "CharacterFamilySecondCoeff"] = round(float(refinement["runner_up_score"]), 1)
        df.at[row_index, "CharacterFamilyMargin"] = round(float(refinement["winner_margin"]), 1)

    for (race_class, player_name), player_rows in df.groupby(["RaceClass", "FixPlayerName"], sort=False):
        if not player_name or str(player_name).startswith("PlayerNameMissing_"):
            continue
        player_row_indices = [row_index for row_index in player_rows.index if row_index in row_refinements]
        if len(player_row_indices) < VARIANT_FAMILY_REFINEMENT_MIN_SUPPORT:
            continue
        family_names = {str(row_refinements[row_index]["family_name"]) for row_index in player_row_indices}
        for family_name in family_names:
            family_row_indices = [
                row_index for row_index in player_row_indices
                if str(row_refinements[row_index]["family_name"]) == family_name
            ]
            if len(family_row_indices) < VARIANT_FAMILY_REFINEMENT_MIN_SUPPORT:
                continue

            winner_counts = defaultdict(int)
            margins_by_winner: dict[str, list[float]] = defaultdict(list)
            for row_index in family_row_indices:
                refinement = row_refinements[row_index]
                winner_name = str(refinement["winner_name"])
                winner_counts[winner_name] += 1
                margins_by_winner[winner_name].append(float(refinement["winner_margin"]))
            dominant_name, dominant_count = max(winner_counts.items(), key=lambda item: (item[1], item[0]))
            dominant_ratio = dominant_count / max(1, len(family_row_indices))
            dominant_avg_margin = sum(margins_by_winner[dominant_name]) / max(1, len(margins_by_winner[dominant_name]))
            if dominant_ratio < VARIANT_FAMILY_REFINEMENT_MIN_DOMINANT_RATIO:
                continue
            if dominant_avg_margin < VARIANT_FAMILY_REFINEMENT_MIN_AVG_MARGIN:
                continue

            dominant_template = template_by_name.get(dominant_name)
            if dominant_template is None:
                continue

            for row_index in family_row_indices:
                refinement = row_refinements[row_index]
                existing_method = str(df.at[row_index, "CharacterMatchMethod"] or "").strip()
                df.at[row_index, "Character"] = dominant_name
                df.at[row_index, "CharacterIndex"] = int(dominant_template["character_index"])
                df.at[row_index, "CharacterMatchConfidence"] = round(float(refinement["winner_score"]), 1)
                df.at[row_index, "CharacterMatchMethod"] = (
                    f"{existing_method}+{VARIANT_FAMILY_REFINEMENT_METHOD}" if existing_method else VARIANT_FAMILY_REFINEMENT_METHOD
                )
    return df
PLAYER_NAME_OCR_ENGINE = os.environ.get("MK8_PLAYER_NAME_OCR_ENGINE", "easyocr").strip().lower()
PLAYER_NAME_BATCH_RAW_MODE = os.environ.get("MK8_PLAYER_NAME_BATCH_RAW_MODE", "weak").strip().lower()
if PLAYER_NAME_BATCH_RAW_MODE not in {"all", "weak", "off"}:
    PLAYER_NAME_BATCH_RAW_MODE = "all"
PLAYER_NAME_EASYOCR_LANGS = [
    part.strip()
    for part in os.environ.get("MK8_PLAYER_NAME_EASYOCR_LANGS", "en,es,fr,de,nl,pl").split(",")
    if part.strip()
]
TRACK_EASYOCR_LANGS = [
    part.strip()
    for part in os.environ.get("MK8_TRACK_EASYOCR_LANGS", "en,nl").split(",")
    if part.strip()
]


def easyocr_gpu_enabled() -> bool:
    runtime_config = load_app_config()
    return runtime_easyocr_gpu_enabled(runtime_config)


def current_ocr_workers() -> int:
    runtime_config = load_app_config()
    return 1 if easyocr_gpu_enabled() else runtime_config.ocr_workers

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class OcrProfiler:
    def __init__(self):
        self._lock = threading.Lock()
        self._stats = defaultdict(lambda: {"calls": 0, "seconds": 0.0})

    def record(self, label: str, duration_s: float) -> None:
        with self._lock:
            self._stats[label]["calls"] += 1
            self._stats[label]["seconds"] += duration_s

    def summary_lines(self) -> List[str]:
        with self._lock:
            stats = {key: value.copy() for key, value in self._stats.items()}
        if not stats:
            return ["OCR engine profile (cumulative across all OCR calls)", "No engine OCR calls recorded"]
        total_calls = sum(item["calls"] for item in stats.values())
        total_seconds = sum(item["seconds"] for item in stats.values())
        rows = []
        for label, item in sorted(stats.items(), key=lambda pair: pair[1]["seconds"], reverse=True):
            avg_ms = (item["seconds"] / max(1, item["calls"])) * 1000.0
            rows.append(
                (
                    str(label),
                    f"{int(item['calls']):>5}",
                    f"{item['seconds']:>10.2f}",
                    f"{avg_ms:>9.1f}",
                )
            )
        headers = ("Method", "Calls", "Total (s)", "Avg (ms)")
        widths = [len(header) for header in headers]
        for row in rows:
            for index, value in enumerate(row):
                widths[index] = max(widths[index], len(value))
        lines = [
            "OCR engine profile (cumulative across all OCR calls)",
            f"Total calls: {total_calls} | Cumulative engine time: {total_seconds:.2f}s",
            "  "
            + "  ".join(
                header.ljust(widths[index]) if index == 0 else header.rjust(widths[index])
                for index, header in enumerate(headers)
            ),
            "  "
            + "  ".join(
                ("-" * widths[index]).ljust(widths[index]) if index == 0 else ("-" * widths[index]).rjust(widths[index])
                for index in range(len(headers))
            ),
        ]
        for row in rows:
            lines.append(
                "  "
                + "  ".join(
                    row[index].ljust(widths[index]) if index == 0 else row[index].rjust(widths[index])
                    for index in range(len(headers))
                )
            )
        return lines


OCR_PROFILER = OcrProfiler()
PLAYER_NAME_FALLBACK_STATS = defaultdict(int)
_EASYOCR_READERS: dict[tuple[str, ...], object] = {}
_EASYOCR_READER_LOCK = threading.Lock()
_EASYOCR_READER_RUN_LOCKS: dict[tuple[str, ...], threading.Lock] = {}
_OCR_TRACE_LOCK = threading.Lock()


def ocr_trace_enabled() -> bool:
    return os.environ.get("MK8_TRACE_OCR_LINKING", "0").strip().lower() in {"1", "true", "yes", "on"}


def ocr_trace_label() -> str:
    return os.environ.get("MK8_OCR_TRACE_LABEL", "").strip() or "adhoc"


def ocr_trace_mode() -> str:
    return os.environ.get("MK8_OCR_TRACE_MODE", "").strip() or "unspecified"


def ocr_trace_root() -> Path:
    return Path(PROJECT_ROOT) / "Output_Results" / "Debug" / "OCR_Tracing" / ocr_trace_label() / ocr_trace_mode()


def _sanitize_trace_part(value: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", str(value or "").strip())
    cleaned = cleaned.strip(" ._")
    return cleaned or "unknown"


def append_ocr_trace_event(filename: str, payload: dict) -> None:
    if not ocr_trace_enabled():
        return
    trace_path = ocr_trace_root() / filename
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    event_payload = dict(payload)
    event_payload.setdefault("trace_label", ocr_trace_label())
    event_payload.setdefault("trace_mode", ocr_trace_mode())
    event_payload.setdefault("pid", os.getpid())
    event_payload.setdefault("ts", time.time())
    with _OCR_TRACE_LOCK:
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event_payload, ensure_ascii=False) + "\n")


def write_ocr_trace_json(relative_path: str, payload: dict) -> None:
    if not ocr_trace_enabled():
        return
    trace_path = ocr_trace_root() / relative_path
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _identity_trace_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "RaceClass",
        "RaceIDNumber",
        "RacePosition",
        "TrackName",
        "PlayerName",
        "FixPlayerName",
        "IdentityLabel",
        "IdentityResolutionMethod",
        "Character",
        "CharacterIndex",
        "DetectedTotalScore",
        "OldTotalScore",
        "NewTotalScore",
        "NameConfidence",
        "NameAllowedCharRatio",
        "NameUnknownChars",
        "NameValidationFlags",
        "ReviewNeeded",
        "ReviewReason",
        "SessionResetDetected",
        "IdentityRelinkDetected",
        "IdentityRelinkSummary",
        "IdentityRelinkNote",
    ]
    trace_df = df.copy()
    for column_name in columns:
        if column_name not in trace_df.columns:
            trace_df[column_name] = ""
    return trace_df[columns].sort_values(["RaceClass", "RaceIDNumber", "RacePosition"], kind="stable")


def write_identity_trace_stage(stage_name: str, df: pd.DataFrame) -> None:
    if not ocr_trace_enabled() or df.empty:
        return
    for race_class, race_group in _identity_trace_columns(df).groupby("RaceClass", sort=False):
        video_dir = _sanitize_trace_part(race_class)
        trace_path = ocr_trace_root() / "identity_stages" / video_dir / f"{stage_name}.csv"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        race_group.to_csv(trace_path, index=False, encoding="utf-8-sig")


def reset_player_name_fallback_stats() -> None:
    PLAYER_NAME_FALLBACK_STATS.clear()


def player_name_fallback_summary_lines() -> List[str]:
    if not PLAYER_NAME_FALLBACK_STATS:
        return []
    total = int(PLAYER_NAME_FALLBACK_STATS.get("fallback_rows", 0))
    return [
        "Player-name fallback activity",
        f"- fallback rows: {total}",
        f"- low batch confidence: {int(PLAYER_NAME_FALLBACK_STATS.get('reason_low_confidence', 0))}",
        f"- short batch text: {int(PLAYER_NAME_FALLBACK_STATS.get('reason_short_text', 0))}",
        f"- low batch character diversity: {int(PLAYER_NAME_FALLBACK_STATS.get('reason_low_diversity', 0))}",
        f"- fallback improved confidence: {int(PLAYER_NAME_FALLBACK_STATS.get('fallback_improved_confidence', 0))}",
        f"- fallback kept/better text length: {int(PLAYER_NAME_FALLBACK_STATS.get('fallback_kept_or_improved_length', 0))}",
    ]


class ProgressPrinter:
    """Print throttled progress updates for OCR/export stages."""

    def __init__(self, scope: str, total_units: int, percent_step: int = 10, min_interval_s: float = 2.0):
        self.scope = scope
        self.total_units = max(1, int(total_units))
        self.percent_step = max(1, int(percent_step))
        self.min_interval_s = float(min_interval_s)
        self.last_percent = -1
        self.last_print_time = 0.0
        self.start_time = time.perf_counter()
        self.phase_peak = {
            "cpu_percent": None,
            "ram_used_gb": None,
            "ram_total_gb": None,
            "gpu_percent": None,
            "vram_used_gb": None,
            "vram_total_gb": None,
        }

    def update(self, completed_units: int, detail: str = "") -> None:
        percent = min(100, int((max(0, completed_units) / self.total_units) * 100))
        now = time.perf_counter()
        should_print = percent >= 100 or self.last_percent < 0
        if not should_print and percent >= self.last_percent + self.percent_step:
            should_print = True
        if not should_print and now - self.last_print_time >= self.min_interval_s:
            should_print = True
        if not should_print:
            return
        detail_suffix = f" | {detail}" if detail else ""
        snapshot = LOGGER.resources.sample()
        self._update_phase_peak(snapshot)
        resource_text = LOGGER.resource_text(snapshot)
        LOGGER.log(
            self.scope,
            f"{completed_units}/{self.total_units} ({percent}%) | {resource_text}{detail_suffix}",
            color_name="magenta",
        )
        self.last_percent = percent
        self.last_print_time = now

    def heartbeat(self, completed_units: int, detail: str = "") -> None:
        now = time.perf_counter()
        if now - self.last_print_time < self.min_interval_s:
            return
        percent = min(100, int((max(0, completed_units) / self.total_units) * 100))
        snapshot = LOGGER.resources.sample()
        self._update_phase_peak(snapshot)
        resource_text = LOGGER.resource_text(snapshot)
        detail_suffix = f" | {detail}" if detail else ""
        LOGGER.log(
            self.scope,
            f"{completed_units}/{self.total_units} ({percent}%) | {resource_text}{detail_suffix}",
            color_name="magenta",
        )
        self.last_print_time = now

    def _update_phase_peak(self, snapshot) -> None:
        for field in ("cpu_percent", "ram_used_gb", "gpu_percent", "vram_used_gb"):
            current = getattr(snapshot, field)
            peak_value = self.phase_peak[field]
            if current is not None and (peak_value is None or current > peak_value):
                self.phase_peak[field] = current
        for field in ("ram_total_gb", "vram_total_gb"):
            current = getattr(snapshot, field)
            if current is not None:
                self.phase_peak[field] = current

    def peak_lines(self) -> list[str]:
        lines = []
        if self.phase_peak["cpu_percent"] is not None:
            lines.append(f"Peak CPU: {self.phase_peak['cpu_percent']:.0f}%")
        if self.phase_peak["ram_used_gb"] is not None and self.phase_peak["ram_total_gb"] is not None:
            lines.append(f"Peak RAM: {self.phase_peak['ram_used_gb']:.1f} / {self.phase_peak['ram_total_gb']:.1f} GB")
        if self.phase_peak["gpu_percent"] is not None:
            lines.append(f"Peak GPU: {self.phase_peak['gpu_percent']:.0f}%")
        if self.phase_peak["vram_used_gb"] is not None and self.phase_peak["vram_total_gb"] is not None:
            lines.append(f"Peak VRAM: {self.phase_peak['vram_used_gb']:.1f} / {self.phase_peak['vram_total_gb']:.1f} GB")
        return lines


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02}"


def pluralize(count: int, singular: str, plural: str | None = None) -> str:
    if count == 1:
        return singular
    return plural or f"{singular}s"

@lru_cache(maxsize=128)
def resolve_video_dimensions(video_value: str, input_videos_folder: str) -> tuple[int | None, int | None]:
    video_path = Path(str(video_value))
    if not video_path.is_absolute():
        video_path = Path(input_videos_folder) / video_path
    if not video_path.exists():
        return None, None
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None, None
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return (width or None, height or None)


@lru_cache(maxsize=128)
def resolve_video_path_for_race_class(race_class: str, input_videos_folder: str) -> str | None:
    input_root = Path(input_videos_folder)
    for video_path in input_root.rglob('*'):
        if not video_path.is_file():
            continue
        if build_video_identity(video_path, input_root=input_root, include_subfolders=True) == race_class:
            return str(video_path)
    return None


def resolve_low_res_metadata(metadata_entry, input_videos_folder: Path, race_class: str) -> tuple[int | None, int | None, bool]:
    video_value = None
    if metadata_entry is not None:
        video_value = str(metadata_entry.get('video', '')).strip() or None
    if not video_value:
        video_value = resolve_video_path_for_race_class(race_class, str(input_videos_folder))
    if not video_value:
        return None, None, False
    width, height = resolve_video_dimensions(video_value, str(input_videos_folder))
    return width, height, is_low_res_height(height, APP_CONFIG.low_res_max_source_height)


def get_race_points(position: int, num_players: int) -> int:
    """Return the points based on position and number of players."""
    # Define point tables for different player counts
    points_table = {
        12: [15, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        11: [13, 11, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        10: [12, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        9: [11, 9, 8, 6, 5, 4, 3, 2, 1, 0],
        8: [10, 8, 6, 5, 4, 3, 2, 1, 0],
        7: [9, 7, 5, 4, 3, 2, 1, 0],
        6: [7, 5, 4, 3, 2, 1, 0],
        5: [6, 4, 3, 2, 1, 0],
        4: [4, 3, 2, 1, 0],
        3: [3, 2, 1, 0],
        2: [2, 1, 0],
    }

    # Default to 12 players if not explicitly defined
    if num_players not in points_table:
        num_players = 12

    # Get the points for the given position
    if 1 <= position <= len(points_table[num_players]):
        return points_table[num_players][position - 1]
    else:
        return 0  # No points for invalid positions


def extract_text_with_confidence(image_source, coordinates: Dict[str, List[Tuple[Tuple[int, int], Tuple[int, int]]]],
                                 lang: str, config: str) -> Tuple[Dict[str, List[str]], List[int]]:
    """Extract text and confidence scores from image ROIs using EasyOCR."""
    extracted_text = {}
    confidence_scores = []
    if isinstance(image_source, str):
        image = cv2.imread(image_source)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {image_source}")
    else:
        image = image_source
        if image is None:
            raise ValueError("Image source array is None")

    for region_type, coord_list in coordinates.items():
        region_text = []
        region_confidence = []
        for (x1, y1), (x2, y2) in coord_list:
            roi = image[y1:y2, x1:x2]
            combined_text, average_confidence = _run_easyocr_single_roi(
                roi,
                TRACK_EASYOCR_LANGS,
                profile_label=f"{region_type}_roi",
            )

            region_text.append(combined_text)
            region_confidence.append(average_confidence)

        extracted_text[region_type] = region_text  # List of combined texts for each ROI
        confidence_scores.extend(region_confidence)

    return extracted_text, confidence_scores

def _get_easyocr_reader(languages: list[str] | tuple[str, ...]):
    cache_key = tuple(languages)
    if cache_key in _EASYOCR_READERS:
        return _EASYOCR_READERS[cache_key]
    if easyocr is None:
        return None
    with _EASYOCR_READER_LOCK:
        if cache_key not in _EASYOCR_READERS:
            _EASYOCR_READERS[cache_key] = easyocr.Reader(list(cache_key), gpu=easyocr_gpu_enabled(), verbose=False)
    return _EASYOCR_READERS[cache_key]


def _get_easyocr_reader_run_lock(languages: list[str] | tuple[str, ...]) -> threading.Lock:
    cache_key = tuple(languages)
    lock = _EASYOCR_READER_RUN_LOCKS.get(cache_key)
    if lock is not None:
        return lock
    with _EASYOCR_READER_LOCK:
        lock = _EASYOCR_READER_RUN_LOCKS.get(cache_key)
        if lock is None:
            lock = threading.Lock()
            _EASYOCR_READER_RUN_LOCKS[cache_key] = lock
    return lock


def _run_easyocr_single_roi(
    roi: np.ndarray,
    languages: list[str] | tuple[str, ...],
    *,
    profile_label: str,
) -> tuple[str, int]:
    reader = _get_easyocr_reader(languages)
    if reader is None or roi.size == 0:
        return "", 0

    reader_run_lock = _get_easyocr_reader_run_lock(languages)
    start_time = time.perf_counter()
    with reader_run_lock:
        result = reader.readtext(roi, detail=1, paragraph=False)
    OCR_PROFILER.record(profile_label, time.perf_counter() - start_time)

    parts = []
    probabilities = []
    for item in result:
        if len(item) < 3:
            continue
        text = str(item[1]).strip()
        probability = float(item[2])
        if not text:
            continue
        parts.append(text)
        probabilities.append(max(0.0, min(1.0, probability)))

    combined_text = " ".join(parts).strip()
    average_confidence = int(round((sum(probabilities) / len(probabilities)) * 100.0)) if probabilities else 0
    return combined_text, average_confidence


def _ocr_words_and_confidence(data: dict) -> tuple[str, int]:
    words = []
    confidences = []
    for index, raw_conf in enumerate(data["conf"]):
        try:
            conf = int(raw_conf)
        except (TypeError, ValueError):
            continue
        text = str(data["text"][index]).strip()
        if conf >= 0 and text:
            words.append(text)
            confidences.append(conf)
    combined_text = " ".join(words).strip()
    average_confidence = int(sum(confidences) // len(confidences)) if confidences else 0
    return combined_text, average_confidence


def _player_name_candidate_score(text: str, confidence: int) -> float:
    visible_chars = visible_name_characters(text)
    visible_len = len(visible_chars)
    if visible_len <= 0:
        return float("-inf")
    diversity = distinct_visible_name_count(text)
    allowed_ratio = allowed_name_char_ratio(text)
    unknown_count = len(unknown_name_chars(text))
    unicode_bonus = sum(1 for char in visible_chars if ord(char) > 127)
    alpha_bonus = sum(1 for char in visible_chars if char.isalpha())
    return (
        float(confidence)
        + (allowed_ratio * 30.0)
        + min(visible_len, 10) * 4.0
        + min(diversity, 8) * 3.0
        + unicode_bonus * 10.0
        + alpha_bonus * 1.5
        - unknown_count * 12.0
    )


def _allowed_unicode_bonus_count(text: str) -> int:
    return sum(1 for char in visible_name_characters(text) if ord(char) > 127 and not unknown_name_chars(char))


def _generate_player_name_fallback_candidates(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    lang: str,
) -> list[tuple[str, int, str]]:
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return []

    candidate_specs: list[tuple[str, np.ndarray, str, str]] = []
    candidate_specs.append(("row_eng", roi, lang, "--psm 7"))

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    upscaled_gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    candidate_specs.append(("row_eng_upscaled", upscaled_gray, lang, "--oem 1 --psm 7"))

    _, otsu_binary = cv2.threshold(upscaled_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted_otsu = 255 - otsu_binary
    candidate_specs.append(("row_eng_inv_otsu", inverted_otsu, lang, "--oem 1 --psm 7"))
    candidate_specs.append(("row_latin_inv_otsu", inverted_otsu, "script/Latin", "--oem 1 --psm 7"))

    focus_x1 = min(max(x1 + 6, 0), max(0, x2 - 1))
    focus_y1 = max(0, y1 - 2)
    if focus_x1 < x2 and focus_y1 < y2:
        focus_roi = image[focus_y1:y2, focus_x1:x2]
        if focus_roi.size > 0:
            focus_gray = cv2.cvtColor(focus_roi, cv2.COLOR_BGR2GRAY)
            focus_upscaled = cv2.resize(focus_gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            _, focus_otsu = cv2.threshold(focus_upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            focus_inverted_otsu = 255 - focus_otsu
            candidate_specs.append(("row_eng_focus_inv_otsu", focus_inverted_otsu, lang, "--oem 1 --psm 7"))
            candidate_specs.append(("row_latin_focus_inv_otsu", focus_inverted_otsu, "script/Latin", "--oem 1 --psm 7"))

    candidates: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int]] = set()
    for label, candidate_image, candidate_lang, candidate_config in candidate_specs:
        candidate_text, candidate_confidence = _run_easyocr_player_name(
            candidate_image if candidate_image.ndim == 3 else cv2.cvtColor(candidate_image, cv2.COLOR_GRAY2BGR),
            0,
            0,
            candidate_image.shape[1],
            candidate_image.shape[0],
        )
        key = (candidate_text, candidate_confidence)
        if not candidate_text or key in seen:
            continue
        seen.add(key)
        candidates.append((candidate_text, candidate_confidence, label))
    return candidates


def _run_easyocr_player_name(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> tuple[str, int]:
    return _run_easyocr_player_name_for_context(image, x1, y1, x2, y2, bundle_kind="", field_name="", method_prefix="row_fallback")


def _run_easyocr_player_name_for_context(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    bundle_kind: str,
    field_name: str,
    method_prefix: str,
) -> tuple[str, int]:
    reader = _get_easyocr_reader(PLAYER_NAME_EASYOCR_LANGS)
    if reader is None:
        return "", 0
    reader_run_lock = _get_easyocr_reader_run_lock(PLAYER_NAME_EASYOCR_LANGS)

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return "", 0

    candidates: list[tuple[str, int, float]] = []

    def add_candidate(source_image: np.ndarray, label: str, method_name: str) -> None:
        start_time = time.perf_counter()
        with reader_run_lock:
            result = reader.readtext(source_image, detail=1, paragraph=False)
        duration_s = time.perf_counter() - start_time
        OCR_PROFILER.record(label, duration_s)
        if bundle_kind and field_name:
            record_call_matrix(bundle_kind, field_name, method_name, duration_s)
        parts = []
        probabilities = []
        for item in result:
            if len(item) < 3:
                continue
            text = str(item[1]).strip()
            probability = float(item[2])
            if not text:
                continue
            parts.append(text)
            probabilities.append(max(0.0, min(1.0, probability)))
        combined_text = " ".join(parts).strip()
        if not combined_text:
            return
        average_confidence = int(round((sum(probabilities) / len(probabilities)) * 100.0)) if probabilities else 0
        candidate_score = _player_name_candidate_score(combined_text, average_confidence)
        candidates.append((combined_text, average_confidence, candidate_score))

    add_candidate(roi, "player_name_easyocr_roi", f"{method_prefix}_raw")

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    upscaled_gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, otsu_binary = cv2.threshold(upscaled_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted_otsu = 255 - otsu_binary
    add_candidate(inverted_otsu, "player_name_easyocr_inv_otsu", f"{method_prefix}_inv_otsu")

    if not candidates:
        return "", 0
    best_text, best_confidence, _best_score = max(candidates, key=lambda item: item[2])
    return best_text, best_confidence


def _build_player_name_canvas(
    image: np.ndarray,
    coord_list: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    *,
    preprocess: str = "raw",
) -> tuple[np.ndarray, list[list[int]]]:
    rois = []
    widths = []
    heights = []
    for (x1, y1), (x2, y2) in coord_list:
        roi = image[y1:y2, x1:x2]
        if preprocess == "inv_otsu3":
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            upscaled_gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            _, otsu_binary = cv2.threshold(upscaled_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            roi = 255 - otsu_binary
        rois.append(roi)
        heights.append(max(1, roi.shape[0]))
        widths.append(max(1, roi.shape[1]))

    separator_height = PLAYER_NAME_BATCH_SEPARATOR_HEIGHT
    canvas_height = sum(heights) + separator_height * max(0, len(rois) - 1)
    canvas_width = max(widths) if widths else 1
    if preprocess == "raw":
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    else:
        canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

    horizontal_list: list[list[int]] = []
    cursor = 0
    for roi in rois:
        height, width = roi.shape[:2]
        canvas[cursor:cursor + height, 0:width] = roi
        horizontal_list.append([0, width, cursor, cursor + height])
        cursor += height + separator_height
    return canvas, horizontal_list


def _run_easyocr_player_names_batched_variant(
    image: np.ndarray,
    coord_list: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    *,
    preprocess: str,
    bundle_kind: str = "",
    field_name: str = "",
) -> tuple[list[str], list[int]]:
    reader = _get_easyocr_reader(PLAYER_NAME_EASYOCR_LANGS)
    if reader is None:
        return [""] * len(coord_list), [0] * len(coord_list)

    reader_run_lock = _get_easyocr_reader_run_lock(PLAYER_NAME_EASYOCR_LANGS)
    canvas, horizontal_list = _build_player_name_canvas(image, coord_list, preprocess=preprocess)
    start_time = time.perf_counter()
    with reader_run_lock:
        result = reader.recognize(
            canvas,
            horizontal_list=horizontal_list,
            free_list=[],
            detail=1,
            paragraph=False,
            reformat=True,
        )
    duration_s = time.perf_counter() - start_time
    profile_label = "player_name_easyocr_batch_raw" if preprocess == "raw" else "player_name_easyocr_batch_inv_otsu"
    OCR_PROFILER.record(profile_label, duration_s)
    if bundle_kind and field_name:
        method_name = "batch_raw" if preprocess == "raw" else "batch_inv_otsu"
        record_call_matrix(bundle_kind, field_name, method_name, duration_s)

    extracted_names = [""] * len(coord_list)
    confidence_scores = [0] * len(coord_list)
    for row_index, item in enumerate(result):
        if len(item) < 3:
            continue
        text = str(item[1]).strip()
        probability = float(item[2])
        if not text:
            continue
        extracted_names[row_index] = text
        confidence_scores[row_index] = int(round(max(0.0, min(1.0, probability)) * 100.0))
    return extracted_names, confidence_scores


def _run_easyocr_player_names_batched(
    image: np.ndarray,
    coord_list: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    *,
    bundle_kind: str,
    field_name: str,
) -> tuple[list[str], list[int]]:
    reader = _get_easyocr_reader(PLAYER_NAME_EASYOCR_LANGS)
    if reader is None:
        return [""] * len(coord_list), [0] * len(coord_list)

    per_row_candidates: list[list[tuple[str, int]]] = [[] for _ in coord_list]
    inv_names, inv_confidences = _run_easyocr_player_names_batched_variant(
        image,
        coord_list,
        preprocess="inv_otsu3",
        bundle_kind=bundle_kind,
        field_name=field_name,
    )
    for row_index, (text, confidence) in enumerate(zip(inv_names, inv_confidences)):
        if not text:
            continue
        per_row_candidates[row_index].append((text, confidence))

    raw_mode = PLAYER_NAME_BATCH_RAW_MODE
    should_run_raw = raw_mode == "all"
    if raw_mode == "weak":
        should_run_raw = any(
            confidence < PLAYER_NAME_BATCH_FALLBACK_CONFIDENCE
            or visible_name_length(text) < 3
            or distinct_visible_name_count(text) < 3
            for text, confidence in zip(inv_names, inv_confidences)
        )

    if should_run_raw:
        raw_names, raw_confidences = _run_easyocr_player_names_batched_variant(
            image,
            coord_list,
            preprocess="raw",
            bundle_kind=bundle_kind,
            field_name=field_name,
        )
        for row_index, (text, confidence) in enumerate(zip(raw_names, raw_confidences)):
            if not text:
                continue
            per_row_candidates[row_index].append((text, confidence))

    extracted_names = []
    confidence_scores = []
    for row_candidates in per_row_candidates:
        if not row_candidates:
            extracted_names.append("")
            confidence_scores.append(0)
            continue
        best_text, best_confidence = max(
            row_candidates,
            key=lambda item: _player_name_candidate_score(item[0], item[1]),
        )
        extracted_names.append(best_text)
        confidence_scores.append(best_confidence)
    return extracted_names, confidence_scores


def extract_player_names_batched(
    image: np.ndarray,
    coord_list: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    lang: str = "eng",
    config: str = PLAYER_NAME_BATCH_CONFIG,
    *,
    bundle_kind: str = "",
    field_name: str = "",
) -> Tuple[List[str], List[int]]:
    extracted_names: list[str] = []
    confidence_scores: list[int] = []
    texts_by_row: list[list[str]] = []
    confidences_by_row: list[list[int]] = []

    if PLAYER_NAME_OCR_ENGINE == "easyocr":
        easyocr_names, easyocr_confidences = _run_easyocr_player_names_batched(
            image,
            coord_list,
            bundle_kind=bundle_kind,
            field_name=field_name,
        )
        bundle_fallback_enabled = True
        if bundle_kind == "2RaceScore":
            bundle_fallback_enabled = RACE_SCORE_NAME_ROW_FALLBACK_ENABLED
        elif bundle_kind == "3TotalScore":
            bundle_fallback_enabled = TOTAL_SCORE_NAME_ROW_FALLBACK_ENABLED
        for ((x1, y1), (x2, y2)), easyocr_text, easyocr_confidence in zip(coord_list, easyocr_names, easyocr_confidences):
            weak_candidate = (
                easyocr_confidence < PLAYER_NAME_BATCH_FALLBACK_CONFIDENCE
                or visible_name_length(easyocr_text) < 3
                or distinct_visible_name_count(easyocr_text) < 3
            )
            if weak_candidate and bundle_fallback_enabled:
                refined_text, refined_confidence = _run_easyocr_player_name_for_context(
                    image,
                    x1,
                    y1,
                    x2,
                    y2,
                    bundle_kind=bundle_kind,
                    field_name=field_name,
                    method_prefix="row_fallback",
                )
                if _player_name_candidate_score(refined_text, refined_confidence) > _player_name_candidate_score(easyocr_text, easyocr_confidence):
                    easyocr_text = refined_text
                    easyocr_confidence = refined_confidence
            texts_by_row.append([easyocr_text] if easyocr_text else [])
            confidences_by_row.append([easyocr_confidence] if easyocr_text else [])
    else:
        rois = []
        widths = []
        heights = []
        for (x1, y1), (x2, y2) in coord_list:
            roi = image[y1:y2, x1:x2]
            rois.append(roi)
            heights.append(max(1, roi.shape[0]))
            widths.append(max(1, roi.shape[1]))

        separator_height = PLAYER_NAME_BATCH_SEPARATOR_HEIGHT
        horizontal_padding = PLAYER_NAME_BATCH_HORIZONTAL_PADDING
        vertical_padding = PLAYER_NAME_BATCH_VERTICAL_PADDING
        canvas_width = max(widths) + horizontal_padding * 2
        canvas_height = sum(height + vertical_padding * 2 for height in heights) + separator_height * (len(rois) - 1)
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        row_ranges = []
        cursor = 0
        for roi in rois:
            height, width = roi.shape[:2]
            start_y = cursor + vertical_padding
            end_y = start_y + height
            start_x = horizontal_padding
            end_x = start_x + width
            canvas[start_y:end_y, start_x:end_x] = roi
            row_ranges.append((start_y, end_y))
            cursor = end_y + vertical_padding + separator_height

        texts_by_row = [[] for _ in rois]
        confidences_by_row = [[] for _ in rois]

    for row_index, ((x1, y1), (x2, y2)) in enumerate(coord_list):
        combined_text = " ".join(texts_by_row[row_index]).strip()
        average_confidence = int(sum(confidences_by_row[row_index]) // len(confidences_by_row[row_index])) if confidences_by_row[row_index] else 0

        low_confidence = average_confidence < PLAYER_NAME_BATCH_FALLBACK_CONFIDENCE
        short_text = visible_name_length(combined_text) < 3
        low_diversity = distinct_visible_name_count(combined_text) < 3
        if PLAYER_NAME_OCR_ENGINE != "easyocr" and PLAYER_NAME_ROW_FALLBACK_ENABLED and (low_confidence or short_text or low_diversity):
            PLAYER_NAME_FALLBACK_STATS["fallback_rows"] += 1
            if low_confidence:
                PLAYER_NAME_FALLBACK_STATS["reason_low_confidence"] += 1
            if short_text:
                PLAYER_NAME_FALLBACK_STATS["reason_short_text"] += 1
            if low_diversity:
                PLAYER_NAME_FALLBACK_STATS["reason_low_diversity"] += 1
            original_text = combined_text
            original_confidence = average_confidence
            fallback_candidates = _generate_player_name_fallback_candidates(image, x1, y1, x2, y2, lang)
            best_text = combined_text
            best_confidence = average_confidence
            best_score = _player_name_candidate_score(best_text, best_confidence)
            for candidate_text, candidate_confidence, _candidate_label in fallback_candidates:
                candidate_score = _player_name_candidate_score(candidate_text, candidate_confidence)
                if candidate_score > best_score:
                    best_text = candidate_text
                    best_confidence = candidate_confidence
                    best_score = candidate_score
            if average_confidence < PLAYER_NAME_BATCH_FALLBACK_CONFIDENCE:
                preferred_unicode_candidate = None
                preferred_unicode_key = None
                for candidate_text, candidate_confidence, _candidate_label in fallback_candidates:
                    unicode_bonus = _allowed_unicode_bonus_count(candidate_text)
                    if unicode_bonus <= 0:
                        continue
                    candidate_key = (
                        unicode_bonus,
                        allowed_name_char_ratio(candidate_text),
                        visible_name_length(candidate_text),
                        candidate_confidence,
                    )
                    if preferred_unicode_key is None or candidate_key > preferred_unicode_key:
                        preferred_unicode_candidate = (candidate_text, candidate_confidence)
                        preferred_unicode_key = candidate_key
                if preferred_unicode_candidate is not None and _allowed_unicode_bonus_count(best_text) == 0:
                    unicode_text, unicode_confidence = preferred_unicode_candidate
                    if visible_name_length(unicode_text) >= max(2, visible_name_length(best_text) - 2):
                        best_text = unicode_text
                        best_confidence = unicode_confidence
            combined_text = best_text
            average_confidence = best_confidence
            if average_confidence > original_confidence:
                PLAYER_NAME_FALLBACK_STATS["fallback_improved_confidence"] += 1
            if visible_name_length(combined_text) >= visible_name_length(original_text):
                PLAYER_NAME_FALLBACK_STATS["fallback_kept_or_improved_length"] += 1

        extracted_names.append(combined_text)
        confidence_scores.append(average_confidence)

    return extracted_names, confidence_scores


tracks_list = load_track_tuples()

def match_track_name(track_name: str, track_list: List[Tuple[int, str, str, int, str]]) -> str:
    """Match a track name (in English or Dutch) to its English equivalent."""
    best_match_score = 0
    best_match_english_name = track_name  # Default to input if no good match is found

    # Iterate through tracks_list
    for track in track_list:
        english_name = track[1]
        dutch_name = track[2]

        # Compute similarity scores
        score_english = difflib.SequenceMatcher(None, track_name, english_name).ratio()
        score_dutch = difflib.SequenceMatcher(None, track_name, dutch_name).ratio()

        # Debugging output
        #print(f"Checking track: {track_name}")
        #print(f"Against English: {english_name}, Score: {score_english}")
        #print(f"Against Dutch: {dutch_name}, Score: {score_dutch}")

        # Update the best match if the current score is higher
        if score_english > best_match_score:
            best_match_score = score_english
            best_match_english_name = english_name

        if score_dutch > best_match_score:
            best_match_score = score_dutch
            best_match_english_name = english_name

    # Debugging final match
    return best_match_english_name


def get_cup_name(track_name: str, track_list: List[Tuple[int, str, str, int, str]]) -> str:
    """Get the Cup Name corresponding to a track."""
    for track in track_list:
        if track_name == track[1]:  # Match English name
            return track[4]
    return ""


# Use the raw anchor-frame template score here, not the exported consensus vote ratio.
# In practice Mii lookalikes can still cluster around ~79 with extremely small winner margins.
MII_FALLBACK_MAX_CONFIDENCE = 80.0
MII_FALLBACK_MAX_MARGIN = 1.0
MII_FALLBACK_MIN_SUSPECT_RACES = 3
MII_FALLBACK_MIN_DISTINCT_WINNERS = 2
MII_CHARACTER_NAME = "Mii"
MII_CHARACTER_INDEX = 80
MII_CHARACTER_METHOD = "mii_fallback_unstable_non_mii_matches"


def annotate_raw_character_match_metrics(df: pd.DataFrame, frames_folder: str | Path) -> pd.DataFrame:
    """Attach raw best-match and top-2 margin metrics from the saved RaceScore frames."""
    if df.empty:
        return df

    df = df.copy()
    if "CharacterMatchRawBest" not in df.columns:
        df["CharacterMatchRawBest"] = np.nan
    if "CharacterMatchRawMargin" not in df.columns:
        df["CharacterMatchRawMargin"] = np.nan

    templates = load_character_templates()
    if not templates:
        return df

    frame_cache: dict[tuple[str, int], tuple[np.ndarray | None, str]] = {}
    frames_root = Path(frames_folder)
    for row_index, row in df.iterrows():
        race_class = str(row.get("RaceClass", "") or "")
        race_id = int(row.get("RaceIDNumber", 0) or 0)
        position = int(row.get("RacePosition", 0) or 0)
        if not race_class or race_id <= 0 or position <= 0:
            continue
        cache_key = (race_class, race_id)
        if cache_key not in frame_cache:
            preferred_frame = find_score_bundle_anchor_path(race_class, race_id, "2RaceScore")
            if preferred_frame is None:
                frame_cache[cache_key] = (None, "")
            else:
                frame_path = preferred_frame
                frame_cache[cache_key] = (cv2.imread(str(frame_path), cv2.IMREAD_COLOR), score_layout_id_from_filename(frame_path))
        frame_image, score_layout_id = frame_cache[cache_key]
        if frame_image is None:
            continue
        (x1, y1), (x2, y2) = character_row_roi(position - 1, score_layout_id=score_layout_id)
        row_roi = frame_image[y1:y2, x1:x2]
        if row_roi.size == 0:
            continue
        matches = best_character_matches(row_roi, templates, limit=2)
        if not matches:
            continue
        best_confidence = float(matches[0].get("CharacterMatchConfidence", 0.0))
        second_confidence = float(matches[1].get("CharacterMatchConfidence", 0.0)) if len(matches) > 1 else 0.0
        df.at[row_index, "CharacterMatchRawBest"] = round(best_confidence, 1)
        df.at[row_index, "CharacterMatchRawMargin"] = round(best_confidence - second_confidence, 1)
    return df


def _masked_chroma_variant_score(source_image: np.ndarray, template_image: np.ndarray, template_alpha: np.ndarray) -> float:
    if source_image.size == 0:
        return 0.0
    template_height, template_width = template_image.shape[:2]
    if source_image.shape[0] != template_height or source_image.shape[1] != template_width:
        source_image = cv2.resize(source_image, (template_width, template_height), interpolation=cv2.INTER_LINEAR)
    visible_mask = template_alpha > 16
    if int(np.count_nonzero(visible_mask)) <= 0:
        return 0.0

    source_lab = cv2.cvtColor(source_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    template_lab = cv2.cvtColor(template_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    source_hsv = cv2.cvtColor(source_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    template_hsv = cv2.cvtColor(template_image, cv2.COLOR_BGR2HSV).astype(np.float32)

    ab_diff = np.mean(np.abs(source_lab[:, :, 1:3][visible_mask] - template_lab[:, :, 1:3][visible_mask]))
    sat_diff = np.mean(np.abs(source_hsv[:, :, 1][visible_mask] - template_hsv[:, :, 1][visible_mask]))
    val_diff = np.mean(np.abs(source_hsv[:, :, 2][visible_mask] - template_hsv[:, :, 2][visible_mask]))

    source_hue = source_hsv[:, :, 0][visible_mask]
    template_hue = template_hsv[:, :, 0][visible_mask]
    source_sat = source_hsv[:, :, 1][visible_mask]
    template_sat = template_hsv[:, :, 1][visible_mask]
    hue_weight = np.maximum(source_sat, template_sat) / 255.0
    hue_delta = np.abs(source_hue - template_hue)
    hue_delta = np.minimum(hue_delta, 180.0 - hue_delta)
    weighted_hue = float(np.sum(hue_delta * hue_weight) / max(1e-6, float(np.sum(hue_weight)))) if np.any(hue_weight > 0) else 90.0

    score = 1.0
    score -= 0.45 * min(1.0, float(ab_diff) / 255.0)
    score -= 0.25 * min(1.0, float(sat_diff) / 255.0)
    score -= 0.20 * min(1.0, float(weighted_hue) / 90.0)
    score -= 0.10 * min(1.0, float(val_diff) / 255.0)
    return max(0.0, min(1.0, float(score)))


def refine_black_blue_character_variants(df: pd.DataFrame, frames_folder: str | Path) -> pd.DataFrame:
    """Refine black/blue variant pairs using chroma-heavy matching on the raw row ROI."""
    if df.empty or "Character" not in df.columns:
        return df

    df = df.copy()
    templates = load_character_templates()
    template_by_name = {str(template["character_name"]): template for template in templates}
    family_by_name = {
        variant_name: family_name
        for family_name, variant_names in BLACK_BLUE_VARIANT_FAMILIES.items()
        for variant_name in variant_names
    }
    frame_cache: dict[tuple[str, int], tuple[np.ndarray | None, str]] = {}
    for row_index, row in df.iterrows():
        current_character = str(row.get("Character") or "").strip()
        if current_character not in family_by_name:
            continue
        race_class = str(row.get("RaceClass", "") or "")
        race_id = int(row.get("RaceIDNumber", 0) or 0)
        position = int(row.get("RacePosition", 0) or 0)
        if not race_class or race_id <= 0 or position <= 0:
            continue
        current_template = template_by_name.get(current_character)
        if current_template is None:
            continue
        family_name = family_by_name[current_character]
        alternate_character = next(
            (candidate for candidate in BLACK_BLUE_VARIANT_FAMILIES[family_name] if candidate != current_character),
            "",
        )
        alternate_template = template_by_name.get(alternate_character)
        if alternate_template is None:
            continue

        cache_key = (race_class, race_id)
        if cache_key not in frame_cache:
            preferred_frame = find_score_bundle_anchor_path(race_class, race_id, "2RaceScore")
            if preferred_frame is None:
                frame_cache[cache_key] = (None, "")
            else:
                frame_cache[cache_key] = (cv2.imread(str(preferred_frame), cv2.IMREAD_COLOR), score_layout_id_from_filename(preferred_frame))
        frame_image, score_layout_id = frame_cache[cache_key]
        if frame_image is None:
            continue
        (x1, y1), (x2, y2) = character_row_roi(position - 1, score_layout_id=score_layout_id)
        row_roi = frame_image[y1:y2, x1:x2]
        if row_roi.size == 0:
            continue

        current_score = _masked_chroma_variant_score(
            row_roi,
            current_template["template_image"],
            current_template["template_alpha"],
        )
        alternate_score = _masked_chroma_variant_score(
            row_roi,
            alternate_template["template_image"],
            alternate_template["template_alpha"],
        )
        if alternate_score < BLACK_BLUE_VARIANT_MIN_SCORE:
            continue
        if alternate_score <= current_score + BLACK_BLUE_VARIANT_MARGIN:
            continue

        df.at[row_index, "Character"] = alternate_character
        df.at[row_index, "CharacterIndex"] = alternate_template["character_index"]
        df.at[row_index, "CharacterMatchConfidence"] = round(alternate_score * 100.0, 1)
        existing_method = str(df.at[row_index, "CharacterMatchMethod"] or "").strip()
        suffix = "black_blue_chroma_refine"
        df.at[row_index, "CharacterMatchMethod"] = f"{existing_method}+{suffix}" if existing_method else suffix
    return df


def rescue_placeholder_identity_names(df: pd.DataFrame) -> pd.DataFrame:
    """Replace placeholder identity labels when multi-race row OCR yields a stable real name."""
    if df.empty or "FixPlayerName" not in df.columns:
        return df

    df = df.copy()
    frame_cache: dict[tuple[str, int], np.ndarray | None] = {}
    existing_names_by_video: dict[str, set[str]] = {}
    for race_class, race_group in df.groupby("RaceClass", sort=False):
        existing_names_by_video[race_class] = {
            str(value).strip()
            for value in race_group["FixPlayerName"].dropna().tolist()
            if str(value).strip() and not str(value).startswith(PLACEHOLDER_NAME_PREFIX)
        }

    for (race_class, placeholder_name), player_rows in df.groupby(["RaceClass", "FixPlayerName"], sort=False):
        placeholder_name = str(placeholder_name or "").strip()
        if not placeholder_name.startswith(PLACEHOLDER_NAME_PREFIX):
            continue

        grouped_candidates: dict[str, dict[str, object]] = {}
        for _, row in player_rows.sort_values(["RaceIDNumber", "RacePosition"], kind="stable").iterrows():
            race_id = int(row.get("RaceIDNumber", 0) or 0)
            position = int(row.get("RacePosition", 0) or 0)
            if race_id <= 0 or position <= 0:
                continue
            cache_key = (race_class, race_id)
            if cache_key not in frame_cache:
                anchor_path = find_score_bundle_anchor_path(race_class, race_id, "2RaceScore")
                frame_cache[cache_key] = cv2.imread(str(anchor_path), cv2.IMREAD_COLOR) if anchor_path is not None else None
            frame_image = frame_cache[cache_key]
            if frame_image is None:
                continue
            score_layout_id = str(row.get("ScoreLayoutId") or "").strip() or "lan2_split_2p"
            player_name_coords = get_score_layout(score_layout_id).player_name_coords
            if position - 1 >= len(player_name_coords):
                continue
            (x1, y1), (x2, y2) = player_name_coords[position - 1]
            row_candidates = _generate_player_name_fallback_candidates(frame_image, x1, y1, x2, y2, "eng")
            best_text = ""
            best_confidence = 0
            best_score = float("-inf")
            for candidate_text, candidate_confidence, _candidate_label in row_candidates:
                candidate_text = str(candidate_text or "").strip()
                if not candidate_text:
                    continue
                if candidate_text.startswith(PLACEHOLDER_NAME_PREFIX):
                    continue
                if visible_name_length(candidate_text) < 3 or distinct_visible_name_count(candidate_text) < 3:
                    continue
                candidate_score = _player_name_candidate_score(candidate_text, candidate_confidence)
                if candidate_score > best_score:
                    best_text = candidate_text
                    best_confidence = int(candidate_confidence)
                    best_score = float(candidate_score)
            if not best_text:
                continue
            normalized = preprocess_name(best_text)
            if not normalized:
                continue
            candidate_entry = grouped_candidates.setdefault(
                normalized,
                {"texts": [], "scores": [], "confidences": [], "race_ids": set()},
            )
            candidate_entry["texts"].append(best_text)
            candidate_entry["scores"].append(best_score)
            candidate_entry["confidences"].append(best_confidence)
            candidate_entry["race_ids"].add(race_id)

        if not grouped_candidates:
            continue

        ranked = sorted(
            grouped_candidates.values(),
            key=lambda entry: (
                len(entry["race_ids"]),
                sum(float(score) for score in entry["scores"]),
                choose_canonical_name(entry["texts"]),
            ),
            reverse=True,
        )
        best_entry = ranked[0]
        second_score = sum(float(score) for score in ranked[1]["scores"]) if len(ranked) > 1 else float("-inf")
        best_total_score = sum(float(score) for score in best_entry["scores"])
        best_name = choose_canonical_name(best_entry["texts"])
        if not best_name:
            continue
        best_support = len(best_entry["race_ids"])
        best_average_score = best_total_score / max(1, len(best_entry["scores"]))
        support_ok = best_support >= PLACEHOLDER_RESCUE_MIN_SUPPORT
        score_ok = best_average_score >= PLACEHOLDER_RESCUE_MIN_ROW_SCORE
        margin_ok = second_score == float("-inf") or (best_total_score - second_score) >= PLACEHOLDER_RESCUE_MIN_MARGIN
        unique_ok = best_name not in existing_names_by_video.get(race_class, set())
        replace_allowed = support_ok and score_ok and margin_ok and unique_ok
        forced_choice_allowed = (
            (not replace_allowed)
            and unique_ok
            and (
                (
                    best_support >= PLACEHOLDER_FORCED_CHOICE_MIN_SUPPORT
                    and best_average_score >= PLACEHOLDER_FORCED_CHOICE_MIN_ROW_SCORE
                )
                or (
                    best_support >= 1
                    and best_average_score >= PLACEHOLDER_FORCED_SINGLE_HIT_MIN_ROW_SCORE
                )
            )
        )

        review_note = (
            f'placeholder_name_candidate="{best_name}"'
            f' support={best_support}'
            f' avg_score={best_average_score:.1f}'
        )
        if second_score != float("-inf"):
            review_note += f' margin={best_total_score - second_score:.1f}'
        if forced_choice_allowed:
            review_note += " forced_choice=1"

        replace_mask = (df["RaceClass"] == race_class) & (df["FixPlayerName"] == placeholder_name)
        if not replace_allowed and not forced_choice_allowed:
            if "ReviewReason" in df.columns:
                df.loc[replace_mask, "ReviewReason"] = df.loc[replace_mask, "ReviewReason"].apply(
                    lambda value: (
                        f"{str(value).strip(';')};{review_note}".strip(";")
                        if review_note not in str(value or "")
                        else str(value or "")
                    )
                )
            continue

        df.loc[replace_mask, "FixPlayerName"] = best_name
        if "IdentityLabel" in df.columns:
            df.loc[replace_mask, "IdentityLabel"] = best_name
        if "IdentityResolutionMethod" in df.columns:
            rescue_method = "placeholder_name_forced_choice" if forced_choice_allowed else "placeholder_name_rescue"
            df.loc[replace_mask, "IdentityResolutionMethod"] = df.loc[replace_mask, "IdentityResolutionMethod"].apply(
                lambda value: f"{value}+{rescue_method}" if str(value or "").strip() else rescue_method
            )
        if "ReviewReason" in df.columns:
            df.loc[replace_mask, "ReviewReason"] = df.loc[replace_mask, "ReviewReason"].apply(
                lambda value: (
                    f"{str(value).strip(';')};{review_note}".strip(";")
                    if review_note not in str(value or "")
                    else str(value or "")
                )
            )
        existing_names_by_video.setdefault(race_class, set()).add(best_name)
    return df


def apply_mii_character_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """Replace persistently unstable low-confidence character matches with Mii."""
    if df.empty or "FixPlayerName" not in df.columns:
        return df

    df = df.copy()
    for (race_class, player_name), player_rows in df.groupby(["RaceClass", "FixPlayerName"], sort=False):
        if not player_name or str(player_name).startswith("PlayerNameMissing_"):
            continue

        suspect_indices = []
        distinct_winners = set()
        for row_index, row in player_rows.iterrows():
            try:
                confidence = float(row.get("CharacterMatchRawBest", row.get("CharacterMatchConfidence", 0.0)) or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            try:
                margin = float(row.get("CharacterMatchRawMargin", np.nan))
            except (TypeError, ValueError):
                margin = np.nan
            suspect = confidence <= MII_FALLBACK_MAX_CONFIDENCE and not pd.isna(margin) and margin <= MII_FALLBACK_MAX_MARGIN

            if suspect:
                suspect_indices.append(row_index)
                winner = str(row.get("Character", "") or "").strip()
                if winner and winner != MII_CHARACTER_NAME:
                    distinct_winners.add(winner)

        if len(suspect_indices) < MII_FALLBACK_MIN_SUSPECT_RACES:
            continue
        if len(distinct_winners) < MII_FALLBACK_MIN_DISTINCT_WINNERS:
            continue

        for row_index in player_rows.index:
            existing_method = str(df.at[row_index, "CharacterMatchMethod"] or "").strip()
            df.at[row_index, "Character"] = MII_CHARACTER_NAME
            df.at[row_index, "CharacterIndex"] = MII_CHARACTER_INDEX
            df.at[row_index, "CharacterMatchMethod"] = (
                f"{existing_method}+{MII_CHARACTER_METHOD}" if existing_method else MII_CHARACTER_METHOD
            )
    return df

def process_race_group(grouped_item, text_detected_folder, metadata_index, input_videos_folder, in_memory_frame_bundles=None):
    """Process a single race group and return extracted rows."""
    race_start_time = time.perf_counter()
    (race_class, race_id_number), images = grouped_item
    if len(images) < 2:
        return {"rows": [], "summary": None, "duration_s": time.perf_counter() - race_start_time}

    track_name_image = None
    race_score_image = None
    total_score_image = None

    for frame_content, image_path in images:
        if frame_content == "0TrackName":
            track_name_image = image_path
        elif frame_content == "2RaceScore":
            race_score_image = image_path
        elif frame_content == "3TotalScore":
            total_score_image = image_path

    if not race_score_image:
        return {"rows": [], "summary": None, "duration_s": time.perf_counter() - race_start_time}

    results = []
    track_name_text = "UNKNOWN"
    if track_name_image:
        track_name_img = cv2.imread(track_name_image)
        coordinates = {"TrackName": [((319, 633), (925, 685))]}
        track_name_data, _ = extract_text_with_confidence(track_name_img, coordinates, 'eng', '--psm 7')

        raw_track_name_text = " ".join(track_name_data['TrackName']).strip()
        track_name_text = match_track_name(raw_track_name_text, tracks_list)

    race_metadata = find_metadata_entry(metadata_index, race_class, race_id_number, "RaceScore")
    total_metadata = find_metadata_entry(metadata_index, race_class, race_id_number, "TotalScore") if total_score_image else None
    score_layout_id = (
        str((race_metadata or {}).get("score_layout_id", "")).strip()
        or score_layout_id_from_filename(race_score_image)
    )

    source_video_width, source_video_height, is_low_res = resolve_low_res_metadata(race_metadata, input_videos_folder, race_class)

    annotate_path = None
    total_annotate_path = None
    annotate_paths = None
    total_annotate_paths = None
    if APP_CONFIG.write_debug_score_images:
        annotate_path = str(debug_score_frame_path(race_class, race_id_number, "2RaceScore"))
        if total_score_image:
            total_annotate_path = str(debug_score_frame_path(race_class, race_id_number, "3TotalScore"))

    race_bundle_key = (race_class, race_id_number, "RaceScore")
    total_bundle_key = (race_class, race_id_number, "TotalScore")
    race_frame_entries = load_consensus_frame_entries(
        race_score_image,
        race_metadata,
        input_videos_folder,
        OCR_CONSENSUS_FRAMES,
        in_memory_frames=(in_memory_frame_bundles or {}).get(race_bundle_key),
    )
    race_frames = [frame for _frame_number, frame in race_frame_entries]
    total_frame_entries = load_consensus_frame_entries(
        total_score_image or race_score_image,
        total_metadata,
        input_videos_folder,
        TOTAL_SCORE_CONSENSUS_WINDOW_SIZE,
        in_memory_frames=(in_memory_frame_bundles or {}).get(total_bundle_key),
    )
    total_frames = [frame for _frame_number, frame in total_frame_entries]
    preselected_points_anchor_frame = None
    preselected_point_frames = None
    preselected_late_frames = None
    race_context_entries = []
    for context_path in find_score_bundle_race_context_paths(race_class, race_id_number, "2RaceScore"):
        frame = cv2.imread(str(context_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        frame_number = extract_exported_frame_number(context_path.stem)
        if frame_number < 0:
            continue
        race_context_entries.append((frame_number, frame))
    if race_context_entries:
        race_context_entries = sorted(race_context_entries, key=lambda item: int(item[0]))
        mid_index = len(race_context_entries) // 2
        preselected_points_anchor_frame = int(race_context_entries[mid_index][0])
        preselected_point_frames = [frame for _frame_number, frame in race_context_entries[:mid_index + 1]]
        preselected_late_frames = [frame for _frame_number, frame in race_context_entries[mid_index:]]
    if APP_CONFIG.write_debug_score_images:
        race_mid_index = len(race_frame_entries) // 2 if race_frame_entries else -1
        annotate_paths = []
        for index, (frame_number, _frame) in enumerate(race_frame_entries):
            annotate_paths.append(
                annotate_path
                if index == race_mid_index
                else str(debug_score_frame_variant_path(race_class, race_id_number, "2RaceScore", frame_number))
            )
        if total_frame_entries:
            total_mid_index = len(total_frame_entries) // 2
            total_annotate_paths = []
            for index, (frame_number, _frame) in enumerate(total_frame_entries):
                total_annotate_paths.append(
                    total_annotate_path
                    if index == total_mid_index
                    else str(debug_score_frame_variant_path(race_class, race_id_number, "3TotalScore", frame_number))
                )
    consensus = build_consensus_observation(
        race_frames,
        total_frames,
        extract_player_names_batched,
        preprocess_name,
        weighted_similarity,
        frame_numbers=[frame_number for frame_number, _frame in race_frame_entries],
        annotate_path=annotate_path,
        total_annotate_path=total_annotate_path,
        annotate_paths=annotate_paths,
        total_annotate_paths=total_annotate_paths,
        video_context=race_class,
        race_id_number=race_id_number,
        is_low_res=is_low_res,
        score_layout_id=score_layout_id,
        preselected_race_point_anchor_frame=preselected_points_anchor_frame,
        preselected_point_frames=preselected_point_frames,
        preselected_late_frames=preselected_late_frames,
    )
    num_players = len(consensus["rows"])
    race_score_players = int(consensus.get("score_visible_rows", num_players))
    total_score_players = int(consensus.get("total_visible_rows", num_players))
    race_warning_messages = build_race_warning_messages(None, race_score_players, total_score_players, consensus["row_count_confidence"])
    for row in consensus["rows"]:
        race_position = row["RacePosition"]
        race_points_fix = get_race_points(race_position, num_players)
        review_reasons = []
        if row["NameConfidence"] < 45:
            review_reasons.append("low_name_confidence")
        if row["DigitConsensus"] < 55:
            review_reasons.append("low_digit_consensus")
        if consensus["row_count_confidence"] < 60:
            review_reasons.append("unstable_row_count")

        results.append([
            race_class,
            race_id_number,
            track_name_text,
            race_position,
            row["PlayerName"],
            row.get("Character", ""),
            row.get("CharacterIndex"),
            row.get("CharacterMatchConfidence", 0.0),
            row.get("CharacterMatchMethod", ""),
            race_points_fix,
            row["DetectedRacePoints"],
            row.get("DetectedRacePointsSource", ""),
            row.get("DetectedOldTotalScore"),
            row.get("DetectedOldTotalScoreSource", ""),
            row["DetectedTotalScore"],
            row.get("DetectedTotalScoreSource", ""),
            row.get("DetectedNewTotalScore"),
            row.get("DetectedNewTotalScoreSource", ""),
            row.get("PositionAfterRace"),
            *[row.get(column_name) for column_name in POSITION_TEMPLATE_COEFF_COLUMNS],
            row["NameConfidence"],
            row.get("NameAllowedCharRatio", 0.0),
            row.get("NameUnknownChars", ""),
            row.get("NameValidationFlags", ""),
            row["DigitConsensus"],
            consensus["row_count_confidence"],
            race_score_players,
            total_score_players,
            consensus.get("legacy_score_visible_rows", race_score_players),
            consensus.get("legacy_total_visible_rows", total_score_players),
            consensus.get("legacy_row_count_confidence", consensus["row_count_confidence"]),
            consensus.get("score_count_votes", ""),
            consensus.get("total_count_votes", ""),
            consensus.get("legacy_score_count_votes", ""),
            consensus.get("legacy_total_count_votes", ""),
            consensus.get("score_row_metrics_summary", ""),
            consensus.get("total_row_metrics_summary", ""),
            consensus.get("race_score_recovery_used", False),
            consensus.get("race_score_recovery_source", ""),
            consensus.get("race_score_recovery_count", race_score_players),
            consensus.get("race_point_anchor_frame"),
            score_layout_id,
            row.get("TotalScoreMappingMethod", ""),
            row.get("TotalScoreMappingScore"),
            row.get("TotalScoreMappingMargin"),
            row.get("TotalScoreNameSimilarity"),
            source_video_width,
            source_video_height,
            is_low_res,
            ";".join(review_reasons),
        ])

    summary = {
        "race_class": race_class,
        "race_id_number": race_id_number,
        "track_name": track_name_text,
        "race_score_players": race_score_players,
        "total_score_players": total_score_players,
        "warning_messages": race_warning_messages,
    }
    if ocr_trace_enabled():
        trace_video_dir = _sanitize_trace_part(race_class)
        trace_relative_path = f"race_outputs/{trace_video_dir}/race_{int(race_id_number):03}.json"
        write_ocr_trace_json(
            trace_relative_path,
            {
                "race_class": race_class,
                "race_id_number": int(race_id_number),
                "track_name": track_name_text,
                "score_layout_id": score_layout_id,
                "source_video_width": source_video_width,
                "source_video_height": source_video_height,
                "is_low_res": bool(is_low_res),
                "race_frame_numbers": [int(frame_number) for frame_number, _frame in race_frame_entries],
                "total_frame_numbers": [int(frame_number) for frame_number, _frame in total_frame_entries],
                "race_point_anchor_frame": consensus.get("race_point_anchor_frame"),
                "race_score_players": race_score_players,
                "total_score_players": total_score_players,
                "row_count_confidence": float(consensus.get("row_count_confidence", 0.0)),
                "rows": [
                    {
                        "race_position": int(row["RacePosition"]),
                        "player_name": str(row.get("PlayerName") or ""),
                        "character": str(row.get("Character") or ""),
                        "character_index": row.get("CharacterIndex"),
                        "detected_total_score": row.get("DetectedTotalScore"),
                        "detected_old_total_score": row.get("DetectedOldTotalScore"),
                        "name_confidence": row.get("NameConfidence"),
                        "name_allowed_char_ratio": row.get("NameAllowedCharRatio"),
                        "name_validation_flags": str(row.get("NameValidationFlags") or ""),
                    }
                    for row in consensus["rows"]
                ],
            },
        )
        append_ocr_trace_event(
            "race_events.jsonl",
            {
                "event": "race_processed",
                "race_class": race_class,
                "race_id_number": int(race_id_number),
                "track_name": track_name_text,
                "row_count": len(consensus["rows"]),
                "race_score_players": race_score_players,
                "total_score_players": total_score_players,
                "duration_s": time.perf_counter() - race_start_time,
            },
        )
    return {"rows": results, "summary": summary, "duration_s": time.perf_counter() - race_start_time}


def build_grouped_race_images(
    folder_path: str,
    *,
    selected_race_classes=None,
) -> list[tuple[tuple[str, int], list[tuple[str, str]]]]:
    grouped_images = {}
    selected_classes = {str(item) for item in selected_race_classes} if selected_race_classes else None
    for race_class, race_id_number, _race_dir in iter_video_race_dirs(folder_path):
        if selected_classes is not None and race_class not in selected_classes:
            continue
        key = (race_class, race_id_number)
        grouped_images[key] = []
        track_name_path = find_anchor_frame_path(race_class, race_id_number, "0TrackName")
        if track_name_path is not None:
            grouped_images[key].append(("0TrackName", str(track_name_path)))
        race_score_path = find_score_bundle_anchor_path(race_class, race_id_number, "2RaceScore")
        if race_score_path is not None:
            grouped_images[key].append(("2RaceScore", str(race_score_path)))
        total_score_path = find_score_bundle_anchor_path(race_class, race_id_number, "3TotalScore")
        if total_score_path is not None:
            grouped_images[key].append(("3TotalScore", str(total_score_path)))
    grouped_items = []
    for key, images in sorted(grouped_images.items(), key=lambda item: item[0]):
        has_race_score = any(frame_content == "2RaceScore" for frame_content, _ in images)
        if not has_race_score:
            continue
        grouped_items.append((key, images))
    return grouped_items


def build_grouped_race_item(
    folder_path: str,
    race_class: str,
    race_id_number: int,
) -> tuple[tuple[str, int], list[tuple[str, str]]]:
    key = (str(race_class), int(race_id_number))
    images: list[tuple[str, str]] = []
    track_name_path = find_anchor_frame_path(race_class, race_id_number, "0TrackName")
    if track_name_path is not None:
        images.append(("0TrackName", str(track_name_path)))
    race_score_path = find_score_bundle_anchor_path(race_class, race_id_number, "2RaceScore")
    if race_score_path is not None:
        images.append(("2RaceScore", str(race_score_path)))
    total_score_path = find_score_bundle_anchor_path(race_class, race_id_number, "3TotalScore")
    if total_score_path is not None:
        images.append(("3TotalScore", str(total_score_path)))
    return key, images


def finalize_ocr_results(
    results: list[list],
    *,
    folder_path: str,
    phase_start_time: float,
    per_video_ocr_durations: dict[str, float],
    progress_peak_lines: list[str],
    ocr_profiler_lines: list[str],
    write_outputs: bool = True,
    emit_logs: bool = True,
):
    df = pd.DataFrame(results, columns=[
        "RaceClass", "RaceIDNumber", "TrackName", "RacePosition", "PlayerName",
        "Character", "CharacterIndex", "CharacterMatchConfidence", "CharacterMatchMethod",
        "RacePoints", "DetectedRacePoints", "DetectedRacePointsSource", "DetectedOldTotalScore", "DetectedOldTotalScoreSource", "DetectedTotalScore", "DetectedTotalScoreSource", "DetectedNewTotalScore", "DetectedNewTotalScoreSource", "PositionAfterRace",
        *POSITION_TEMPLATE_COEFF_COLUMNS,
        "NameConfidence", "NameAllowedCharRatio", "NameUnknownChars", "NameValidationFlags",
        "DigitConsensus", "RowCountConfidence", "RaceScorePlayerCount", "TotalScorePlayerCount",
        "LegacyRaceScorePlayerCount", "LegacyTotalScorePlayerCount", "LegacyRowCountConfidence",
        "RaceScoreCountVotes", "TotalScoreCountVotes", "LegacyRaceScoreCountVotes", "LegacyTotalScoreCountVotes",
        "RaceScoreRowSignals", "TotalScoreRowSignals",
        "RaceScoreRecoveryUsed", "RaceScoreRecoverySource", "RaceScoreRecoveryCount",
        "RacePointsAnchorFrame",
        "ScoreLayoutId",
        "TotalScoreMappingMethod", "TotalScoreMappingScore", "TotalScoreMappingMargin", "TotalScoreNameSimilarity",
        "SourceVideoWidth", "SourceVideoHeight", "IsLowRes", "ReviewReason"
    ])
    df = df.sort_values(["RaceClass", "RaceIDNumber", "RacePosition"], kind="stable").reset_index(drop=True)
    if df.empty:
        if emit_logs:
            LOGGER.log("[OCR - Phase Complete]", "No races were extracted", color_name="yellow")
        return {"duration_s": 0.0, "output_excel_path": None, "per_video_durations": {}, "df": df, "per_video_summary": {}}

    df['CupName'] = df['TrackName'].apply(lambda name: get_cup_name(name, tracks_list))
    df['TrackID'] = df['TrackName'].apply(lambda name: next((track[0] for track in tracks_list if track[1] == name), None))
    write_identity_trace_stage("raw_ocr_input", df)

    low_res_mask = df["IsLowRes"].fillna(False).astype(bool)
    standardized_frames = []
    high_res_df = df.loc[~low_res_mask].copy()
    if not high_res_df.empty:
        standardized_frames.append(standardize_player_names(high_res_df, os.path.join(PROJECT_ROOT, 'Output_Results', 'Debug'), APP_CONFIG.write_debug_linking_excel))
    low_res_df = df.loc[low_res_mask].copy()
    if not low_res_df.empty:
        frames_folder = os.path.join(PROJECT_ROOT, 'Output_Results', 'Frames')
        for race_class, low_res_group in low_res_df.groupby('RaceClass', sort=False):
            standardized_frames.append(
                apply_low_res_identity_pipeline(
                    low_res_group,
                    frames_folder,
                    race_class,
                    write_debug_outputs=APP_CONFIG.write_debug_csv,
                    debug_dir=os.path.join(PROJECT_ROOT, 'Output_Results', 'Debug'),
                )
            )
    df = pd.concat(standardized_frames, ignore_index=True) if standardized_frames else df.copy()
    df = df.sort_values(["RaceClass", "RaceIDNumber", "RacePosition"], kind="stable").reset_index(drop=True)
    write_identity_trace_stage("after_standardize", df)
    df = resolve_duplicate_name_identity_chains(df)
    write_identity_trace_stage("after_duplicate_name_chain_resolution", df)
    df = merge_fragmented_identity_aliases(df)
    write_identity_trace_stage("after_alias_merge", df)
    frames_folder = os.path.join(PROJECT_ROOT, 'Output_Results', 'Frames')
    df = annotate_raw_character_match_metrics(df, frames_folder)
    df = refine_black_blue_character_variants(df, frames_folder)
    df = refine_character_variant_families(df, frames_folder)
    df = apply_mii_character_fallback(df)

    df = apply_session_validation(df, parse_detected_int, exact_total_score_fallback)
    df = reconcile_connection_reset_identities(df)
    df = compact_identity_labels(df)
    df = rescue_placeholder_identity_names(df)
    write_identity_trace_stage("after_connection_reset_relink", df)
    df = df.sort_values(["RaceClass", "RaceIDNumber", "RacePosition"], kind="stable").reset_index(drop=True)
    df = apply_session_validation(df, parse_detected_int, exact_total_score_fallback)
    df = append_identity_relink_review_notes(df)
    df = append_identity_ambiguity_review_notes(df)
    df = apply_temporary_player_drop_scoring_policy(df)
    write_identity_trace_stage("final_identity_export_input", df)

    if write_outputs:
        completion_payload = build_completion_payload(
            df,
            folder_path,
            phase_start_time,
            progress_peak_lines,
            ocr_profiler_lines,
            per_video_ocr_durations,
            build_race_warning_messages,
            pluralize,
            format_duration,
        )
        if emit_logs:
            LOGGER.summary_block("[OCR - Phase Complete]", completion_payload["lines"], color_name="green")
        return {
            "duration_s": completion_payload["duration_s"],
            "output_excel_path": completion_payload["output_excel_path"],
            "race_count": completion_payload["race_count"],
            "per_video_durations": completion_payload["per_video_durations"],
            "per_video_summary": completion_payload["per_video_summary"],
            "df": df,
            "progress_peak_lines": progress_peak_lines,
            "ocr_profiler_lines": ocr_profiler_lines,
        }

    summary_lines, per_video_summary = build_player_count_summary_lines(df, build_race_warning_messages, pluralize)
    return {
        "duration_s": time.time() - phase_start_time,
        "output_excel_path": None,
        "race_count": int(df[["RaceClass", "RaceIDNumber"]].drop_duplicates().shape[0]),
        "per_video_durations": dict(per_video_ocr_durations),
        "per_video_summary": per_video_summary,
        "df": df,
        "progress_peak_lines": progress_peak_lines,
        "ocr_profiler_lines": ocr_profiler_lines,
        "summary_lines": summary_lines,
    }


def process_images_in_folder(
    folder_path: str,
    in_memory_frame_bundles=None,
    selected_race_classes=None,
    metadata_index_override=None,
    *,
    write_outputs: bool = True,
    emit_logs: bool = True,
    progress_callback=None,
):
    phase_start_time = time.time()
    ocr_workers = current_ocr_workers()
    ocr_consensus_frames = load_app_config().ocr_consensus_frames
    reset_observation_stage_stats()
    reset_call_matrix_stats()
    reset_character_shortlist_state()
    reset_player_name_fallback_stats()
    race_dirs = iter_video_race_dirs(folder_path)

    if not race_dirs:
        if emit_logs:
            LOGGER.log("[OCR - Read text from image - Phase Start]", "The Frames folder is empty. Run extraction first.", color_name="red")
        raise RuntimeError("The Frames folder is empty. Run extraction first.")

    base_dir = Path(PROJECT_ROOT)
    text_detected_folder = os.path.join(PROJECT_ROOT, 'Output_Results', 'Debug', 'Score_Frames')
    if APP_CONFIG.write_debug_score_images and not os.path.exists(text_detected_folder):
        os.makedirs(text_detected_folder)

    linking_data_folder = os.path.join(PROJECT_ROOT, 'Output_Results', 'Debug')
    if APP_CONFIG.write_debug_linking_excel and not os.path.exists(linking_data_folder):
        os.makedirs(linking_data_folder)

    metadata_index = metadata_index_override if metadata_index_override is not None else load_exported_frame_metadata(base_dir)
    input_videos_folder = base_dir / "Input_Videos"

    sorted_grouped_images = build_grouped_race_images(folder_path, selected_race_classes=selected_race_classes)
    if progress_callback is not None:
        progress_callback(
            {
                "event": "start",
                "completed": 0,
                "total": len(sorted_grouped_images),
                "elapsed_s": time.time() - phase_start_time,
            }
        )
    if emit_logs:
        LOGGER.log("[OCR - Read text from image - Phase Start]", "", color_name="magenta")
        LOGGER.log(
            "[OCR - Settings]",
            f"OCR workers: {ocr_workers} | Consensus frames: {ocr_consensus_frames} | Input race groups: {len(sorted_grouped_images)}",
            color_name="magenta",
        )
        if selected_race_classes is not None:
            LOGGER.log(
                "[OCR - Settings]",
                f"Selection scope: {len(selected_race_classes)} video classes",
                color_name="magenta",
            )
    results = []
    race_summaries = []
    per_video_ocr_durations = defaultdict(float)
    race_totals_by_class = {}
    for (race_class, _race_id), _images in sorted_grouped_images:
        race_totals_by_class[race_class] = race_totals_by_class.get(race_class, 0) + 1
    progress = ProgressPrinter("[OCR]", len(sorted_grouped_images), percent_step=5, min_interval_s=2.0)

    with ThreadPoolExecutor(max_workers=ocr_workers) as executor:
        # Each race group is independent, so the safest parallelism boundary is one
        # worker per race bundle.
        future_map = {
            executor.submit(
                process_race_group,
                item,
                text_detected_folder,
                metadata_index,
                input_videos_folder,
                in_memory_frame_bundles,
            ): item
            for item in sorted_grouped_images
        }
        pending = set(future_map.keys())
        completed_count = 0
        while pending:
            done, pending = wait(pending, timeout=3.0, return_when=FIRST_COMPLETED)
            if not done:
                if emit_logs:
                    progress.heartbeat(completed_count, f"In flight: {len(pending)} | Still processing OCR races")
                elif progress_callback is not None:
                    progress_callback(
                        {
                            "event": "heartbeat",
                            "completed": completed_count,
                            "total": len(sorted_grouped_images),
                            "pending": len(pending),
                            "elapsed_s": time.time() - phase_start_time,
                            "detail": "Still processing OCR races",
                        }
                    )
                continue
            for future in done:
                completed_count += 1
                race_result = future.result()
                race_class, race_id_number = future_map[future][0]
                matching_summary = race_result["summary"]
                per_video_ocr_durations[race_class] += float(race_result.get("duration_s", 0.0))
                results.extend(race_result["rows"])
                if race_result["summary"] is not None:
                    race_summaries.append(race_result["summary"])
                if emit_logs:
                    progress.update(completed_count, f"In flight: {len(pending)}")
                elif progress_callback is not None:
                    progress_callback(
                        {
                            "event": "progress",
                            "completed": completed_count,
                            "total": len(sorted_grouped_images),
                            "pending": len(pending),
                            "elapsed_s": time.time() - phase_start_time,
                            "video_label": race_class,
                            "race_id": int(race_id_number),
                            "track_name": None if matching_summary is None else matching_summary["track_name"],
                            "race_score_players": None if matching_summary is None else matching_summary["race_score_players"],
                            "total_score_players": None if matching_summary is None else matching_summary["total_score_players"],
                        }
                    )
                if matching_summary is not None:
                    if emit_logs:
                        LOGGER.log(
                            "",
                            f"Video: {race_class} | Race: {race_id_number:03}/{race_totals_by_class.get(race_class, race_id_number):03} | Track: {matching_summary['track_name']}",
                            color_name="magenta",
                        )
                        LOGGER.log(
                            "",
                            f"Players: race score {matching_summary['race_score_players']} | total score {matching_summary['total_score_players']}",
                            color_name="magenta",
                        )
                else:
                    if emit_logs:
                        LOGGER.log(
                            "",
                            f"Video: {race_class} | Race: {race_id_number:03}/{race_totals_by_class.get(race_class, race_id_number):03}",
                            color_name="magenta",
                        )
                if matching_summary is not None:
                    for warning_message in matching_summary["warning_messages"]:
                        if emit_logs:
                            LOGGER.log(
                                f"[OCR - Warning]",
                                f"Video: {race_class} | Race: {race_id_number:03} | Track: {matching_summary['track_name']} | {warning_message}",
                                color_name="yellow",
                            )

    ocr_profiler_lines = (
        call_matrix_summary_lines(colorize=LOGGER.color)
        + OCR_PROFILER.summary_lines()
        + observation_stage_summary_lines()
        + character_shortlist_summary_lines()
        + player_name_fallback_summary_lines()
    )
    return finalize_ocr_results(
        results,
        folder_path=folder_path,
        phase_start_time=phase_start_time,
        per_video_ocr_durations=dict(per_video_ocr_durations),
        progress_peak_lines=progress.peak_lines(),
        ocr_profiler_lines=ocr_profiler_lines,
        write_outputs=write_outputs,
        emit_logs=emit_logs,
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="OCR Mario Kart 8 extracted frames")
    parser.add_argument("--video", help="Process only a specific video filename or race class stem")
    parser.add_argument(
        "--race-class",
        dest="race_classes",
        action="append",
        help="Process only a specific race class identifier; may be supplied multiple times",
    )
    args = parser.parse_args()

    folder_path = os.path.join(PROJECT_ROOT, 'Output_Results', 'Frames')
    selected_race_classes = list(args.race_classes or [])
    if args.video:
        selected_race_classes.append(Path(args.video).stem)
    process_images_in_folder(folder_path, selected_race_classes=selected_race_classes)


if __name__ == "__main__":
    main()
