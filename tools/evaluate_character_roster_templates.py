from __future__ import annotations

import argparse
import csv
import itertools
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from mk8_local_play.extract_text import find_score_bundle_anchor_path, score_layout_id_from_filename
from mk8_local_play.game_catalog import load_game_catalog
from mk8_local_play.ocr_scoreboard_consensus import CHARACTER_TEMPLATE_SIZE
from mk8_local_play.ocr_scoreboard_consensus import character_row_roi


@dataclass(frozen=True)
class TemplateRecord:
    character_index: int
    roster_index: int
    character_name: str
    template_image: np.ndarray
    template_alpha: np.ndarray


@dataclass(frozen=True)
class StageEvaluation:
    stage_name: str
    feature_names: tuple[str, ...]
    weights: tuple[float, ...]
    correct_count: int
    total_count: int
    min_margin: float
    median_margin: float
    mean_margin: float


def load_runtime_roster_templates(
    *,
    asset_dir: Path,
    start_index: int = 0,
    end_index: int = 78,
    template_size: int = CHARACTER_TEMPLATE_SIZE,
) -> list[TemplateRecord]:
    catalog = load_game_catalog()
    catalog_by_index = {int(item.character_index): item for item in catalog.characters}
    templates: list[TemplateRecord] = []

    for character_index in range(int(start_index), int(end_index) + 1):
        metadata = catalog_by_index.get(character_index)
        if metadata is None:
            raise FileNotFoundError(f"Character index {character_index} not found in game catalog.")

        template_path = asset_dir / f"{character_index}.png"
        if not template_path.exists():
            raise FileNotFoundError(f"Template asset not found: {template_path}")

        image = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Could not read template asset: {template_path}")
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        elif image.shape[2] == 3:
            alpha_channel = np.full(image.shape[:2], 255, dtype=np.uint8)
            image = np.dstack((image, alpha_channel))

        resized = cv2.resize(image, (template_size, template_size), interpolation=cv2.INTER_LINEAR)
        templates.append(
            TemplateRecord(
                character_index=character_index,
                roster_index=int(metadata.roster_index),
                character_name=str(metadata.name_uk),
                template_image=resized[:, :, :3],
                template_alpha=resized[:, :, 3],
            )
        )

    return templates


def _visible_mask(alpha: np.ndarray) -> np.ndarray:
    return np.asarray(alpha > 16, dtype=bool)


def _core_visible_mask(alpha: np.ndarray, *, iterations: int = 1) -> np.ndarray:
    mask = np.asarray(alpha > 16, dtype=np.uint8)
    if int(np.count_nonzero(mask)) <= 0:
        return np.zeros_like(mask, dtype=bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=max(0, int(iterations)))
    if int(np.count_nonzero(eroded)) <= 0:
        eroded = mask
    return np.asarray(eroded > 0, dtype=bool)


def _translate_template(
    image: np.ndarray,
    alpha: np.ndarray,
    *,
    dx: int = 0,
    dy: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = alpha.shape[:2]
    matrix = np.float32([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]])
    shifted_image = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    shifted_alpha = cv2.warpAffine(
        alpha,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return shifted_image, shifted_alpha


def masked_color_score(
    source_image: np.ndarray,
    source_alpha: np.ndarray,
    target_image: np.ndarray,
    target_alpha: np.ndarray,
) -> float:
    source_mask = _visible_mask(source_alpha)
    target_mask = _visible_mask(target_alpha)
    overlap_mask = source_mask & target_mask
    if int(np.count_nonzero(overlap_mask)) <= 0:
        return 0.0
    source_rgb = source_image.astype(np.float32)
    target_rgb = target_image.astype(np.float32)
    mean_abs_diff = float(np.mean(np.abs(source_rgb[overlap_mask] - target_rgb[overlap_mask])))
    return max(0.0, 1.0 - (mean_abs_diff / 255.0))


def masked_cutout_color_score(
    source_image: np.ndarray,
    source_alpha: np.ndarray,
    target_image: np.ndarray,
    target_alpha: np.ndarray,
) -> float:
    source_mask = _visible_mask(source_alpha)
    target_mask = _core_visible_mask(target_alpha)
    overlap_mask = source_mask & target_mask
    if int(np.count_nonzero(overlap_mask)) <= 0:
        return 0.0
    source_rgb = source_image.astype(np.float32)
    target_rgb = target_image.astype(np.float32)
    mean_abs_diff = float(np.mean(np.abs(source_rgb[overlap_mask] - target_rgb[overlap_mask])))
    return max(0.0, 1.0 - (mean_abs_diff / 255.0))


def masked_grayscale_score(
    source_image: np.ndarray,
    source_alpha: np.ndarray,
    target_image: np.ndarray,
    target_alpha: np.ndarray,
) -> float:
    source_mask = _visible_mask(source_alpha)
    target_mask = _visible_mask(target_alpha)
    overlap_mask = source_mask & target_mask
    if int(np.count_nonzero(overlap_mask)) <= 0:
        return 0.0
    source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean_abs_diff = float(np.mean(np.abs(source_gray[overlap_mask] - target_gray[overlap_mask])))
    return max(0.0, 1.0 - (mean_abs_diff / 255.0))


def masked_cutout_grayscale_score(
    source_image: np.ndarray,
    source_alpha: np.ndarray,
    target_image: np.ndarray,
    target_alpha: np.ndarray,
) -> float:
    source_mask = _visible_mask(source_alpha)
    target_mask = _core_visible_mask(target_alpha)
    overlap_mask = source_mask & target_mask
    if int(np.count_nonzero(overlap_mask)) <= 0:
        return 0.0
    source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean_abs_diff = float(np.mean(np.abs(source_gray[overlap_mask] - target_gray[overlap_mask])))
    return max(0.0, 1.0 - (mean_abs_diff / 255.0))


def edge_agreement_score(
    source_image: np.ndarray,
    source_alpha: np.ndarray,
    target_image: np.ndarray,
    target_alpha: np.ndarray,
) -> float:
    source_mask = _visible_mask(source_alpha)
    target_mask = _visible_mask(target_alpha)
    source_edges = cv2.Canny(cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY), 80, 160) > 0
    target_edges = cv2.Canny(cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY), 80, 160) > 0
    source_edges &= source_mask
    target_edges &= target_mask
    overlap = int(np.count_nonzero(source_edges & target_edges))
    total = int(np.count_nonzero(source_edges)) + int(np.count_nonzero(target_edges))
    if total <= 0:
        return 0.0
    return float((2.0 * overlap) / total)


def cutout_edge_agreement_score(
    source_image: np.ndarray,
    source_alpha: np.ndarray,
    target_image: np.ndarray,
    target_alpha: np.ndarray,
) -> float:
    source_mask = _visible_mask(source_alpha)
    target_mask = _core_visible_mask(target_alpha)
    source_edges = cv2.Canny(cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY), 80, 160) > 0
    target_edges = cv2.Canny(cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY), 80, 160) > 0
    source_edges &= source_mask & target_mask
    target_edges &= target_mask
    overlap = int(np.count_nonzero(source_edges & target_edges))
    total = int(np.count_nonzero(source_edges)) + int(np.count_nonzero(target_edges))
    if total <= 0:
        return 0.0
    return float((2.0 * overlap) / total)


def aligned_cutout_blend_score(
    source_image: np.ndarray,
    source_alpha: np.ndarray,
    target_image: np.ndarray,
    target_alpha: np.ndarray,
    *,
    max_offset: int = 4,
) -> tuple[float, float, float, int, int]:
    best_blend = -1.0
    best_color = 0.0
    best_edge = 0.0
    best_dx = 0
    best_dy = 0
    max_offset = max(0, int(max_offset))
    for dy in range(-max_offset, max_offset + 1):
        for dx in range(-max_offset, max_offset + 1):
            shifted_image, shifted_alpha = _translate_template(target_image, target_alpha, dx=dx, dy=dy)
            color_score = masked_cutout_color_score(source_image, source_alpha, shifted_image, shifted_alpha)
            edge_score = cutout_edge_agreement_score(source_image, source_alpha, shifted_image, shifted_alpha)
            blend_score = (0.20 * color_score) + (0.80 * edge_score)
            if blend_score > best_blend:
                best_blend = float(blend_score)
                best_color = float(color_score)
                best_edge = float(edge_score)
                best_dx = int(dx)
                best_dy = int(dy)
    return best_blend, best_color, best_edge, best_dx, best_dy


def silhouette_iou_score(source_alpha: np.ndarray, target_alpha: np.ndarray) -> float:
    source_mask = _visible_mask(source_alpha)
    target_mask = _visible_mask(target_alpha)
    union = int(np.count_nonzero(source_mask | target_mask))
    if union <= 0:
        return 0.0
    intersection = int(np.count_nonzero(source_mask & target_mask))
    return float(intersection / union)


def contour_similarity_score(source_alpha: np.ndarray, target_alpha: np.ndarray) -> float:
    source_mask = np.asarray(_visible_mask(source_alpha), dtype=np.uint8)
    target_mask = np.asarray(_visible_mask(target_alpha), dtype=np.uint8)
    source_area = float(np.count_nonzero(source_mask))
    target_area = float(np.count_nonzero(target_mask))
    if source_area <= 0 or target_area <= 0:
        return 0.0

    source_contours, _ = cv2.findContours(source_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    target_contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not source_contours or not target_contours:
        return 0.0

    source_contour = max(source_contours, key=cv2.contourArea)
    target_contour = max(target_contours, key=cv2.contourArea)
    source_perimeter = float(cv2.arcLength(source_contour, True))
    target_perimeter = float(cv2.arcLength(target_contour, True))
    if source_perimeter <= 0 or target_perimeter <= 0:
        return 0.0

    source_area_score = 1.0 - abs(source_area - target_area) / max(source_area, target_area)
    perimeter_score = 1.0 - abs(source_perimeter - target_perimeter) / max(source_perimeter, target_perimeter)

    source_moments = cv2.HuMoments(cv2.moments(source_mask)).flatten()
    target_moments = cv2.HuMoments(cv2.moments(target_mask)).flatten()
    hu_distance = 0.0
    for source_value, target_value in zip(source_moments, target_moments):
        source_log = -math.copysign(1.0, float(source_value)) * math.log10(max(abs(float(source_value)), 1e-12))
        target_log = -math.copysign(1.0, float(target_value)) * math.log10(max(abs(float(target_value)), 1e-12))
        hu_distance += abs(source_log - target_log)
    hu_score = 1.0 / (1.0 + (hu_distance / 12.0))

    score = (0.35 * max(0.0, source_area_score)) + (0.25 * max(0.0, perimeter_score)) + (0.40 * hu_score)
    return max(0.0, min(1.0, float(score)))


def compute_feature_matrices(templates: Sequence[TemplateRecord]) -> dict[str, np.ndarray]:
    template_count = len(templates)
    feature_matrices = {
        "color": np.zeros((template_count, template_count), dtype=np.float32),
        "gray": np.zeros((template_count, template_count), dtype=np.float32),
        "edge": np.zeros((template_count, template_count), dtype=np.float32),
        "cutout_color": np.zeros((template_count, template_count), dtype=np.float32),
        "cutout_gray": np.zeros((template_count, template_count), dtype=np.float32),
        "cutout_edge": np.zeros((template_count, template_count), dtype=np.float32),
        "silhouette": np.zeros((template_count, template_count), dtype=np.float32),
        "contour": np.zeros((template_count, template_count), dtype=np.float32),
    }

    for source_index, source in enumerate(templates):
        for target_index, target in enumerate(templates):
            feature_matrices["color"][source_index, target_index] = masked_color_score(
                source.template_image,
                source.template_alpha,
                target.template_image,
                target.template_alpha,
            )
            feature_matrices["gray"][source_index, target_index] = masked_grayscale_score(
                source.template_image,
                source.template_alpha,
                target.template_image,
                target.template_alpha,
            )
            feature_matrices["edge"][source_index, target_index] = edge_agreement_score(
                source.template_image,
                source.template_alpha,
                target.template_image,
                target.template_alpha,
            )
            feature_matrices["cutout_color"][source_index, target_index] = masked_cutout_color_score(
                source.template_image,
                source.template_alpha,
                target.template_image,
                target.template_alpha,
            )
            feature_matrices["cutout_gray"][source_index, target_index] = masked_cutout_grayscale_score(
                source.template_image,
                source.template_alpha,
                target.template_image,
                target.template_alpha,
            )
            feature_matrices["cutout_edge"][source_index, target_index] = cutout_edge_agreement_score(
                source.template_image,
                source.template_alpha,
                target.template_image,
                target.template_alpha,
            )
            feature_matrices["silhouette"][source_index, target_index] = silhouette_iou_score(
                source.template_alpha,
                target.template_alpha,
            )
            feature_matrices["contour"][source_index, target_index] = contour_similarity_score(
                source.template_alpha,
                target.template_alpha,
            )

    return feature_matrices


def aggregate_feature_matrices(
    feature_matrices: dict[str, np.ndarray],
    feature_names: Sequence[str],
    weights: Sequence[float],
) -> np.ndarray:
    score_matrix = np.zeros_like(next(iter(feature_matrices.values())))
    for feature_name, weight in zip(feature_names, weights):
        score_matrix += np.asarray(feature_matrices[feature_name], dtype=np.float32) * float(weight)
    return score_matrix


def generate_weight_grid(feature_count: int, step: float = 0.05) -> list[tuple[float, ...]]:
    steps = int(round(1.0 / step))
    if feature_count <= 0:
        return []
    if feature_count == 1:
        return [(1.0,)]

    grid: list[tuple[float, ...]] = []
    for combination in itertools.product(range(steps + 1), repeat=feature_count):
        if sum(combination) != steps:
            continue
        grid.append(tuple(round(value * step, 10) for value in combination))
    return grid


def stage_metrics(score_matrix: np.ndarray) -> tuple[int, np.ndarray]:
    matrix = np.asarray(score_matrix, dtype=np.float32)
    diagonal = np.diag(matrix).astype(np.float32)
    other_scores = matrix.copy()
    np.fill_diagonal(other_scores, -np.inf)
    max_other = np.max(other_scores, axis=1)
    margins = diagonal - max_other
    correct_count = int(np.sum(diagonal > max_other))
    return correct_count, margins


def optimize_stage_weights(
    feature_matrices: dict[str, np.ndarray],
    feature_names: Sequence[str],
    *,
    weight_step: float = 0.05,
) -> StageEvaluation:
    if not feature_names:
        raise ValueError("At least one feature name is required.")

    best_result: StageEvaluation | None = None
    best_objective: tuple[float, ...] | None = None
    for weights in generate_weight_grid(len(feature_names), step=weight_step):
        score_matrix = aggregate_feature_matrices(feature_matrices, feature_names, weights)
        correct_count, margins = stage_metrics(score_matrix)
        objective = (
            float(correct_count),
            float(np.min(margins)),
            float(np.median(margins)),
            float(np.mean(margins)),
        )
        if best_objective is None or objective > best_objective:
            best_objective = objective
            best_result = StageEvaluation(
                stage_name="+".join(feature_names),
                feature_names=tuple(feature_names),
                weights=tuple(float(weight) for weight in weights),
                correct_count=correct_count,
                total_count=int(score_matrix.shape[0]),
                min_margin=float(np.min(margins)),
                median_margin=float(np.median(margins)),
                mean_margin=float(np.mean(margins)),
            )

    if best_result is None:
        raise RuntimeError("No weight combination produced a stage evaluation.")
    return best_result


def build_stage_sequence(feature_matrices: dict[str, np.ndarray], *, weight_step: float) -> list[StageEvaluation]:
    ordered_features = ("color", "gray", "edge", "silhouette", "contour")
    stages: list[StageEvaluation] = []
    for stage_index in range(len(ordered_features)):
        stage_features = ordered_features[: stage_index + 1]
        stage_result = optimize_stage_weights(feature_matrices, stage_features, weight_step=weight_step)
        stages.append(stage_result)
    return stages


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_pairwise_rows(
    templates: Sequence[TemplateRecord],
    feature_matrices: dict[str, np.ndarray],
    final_stage: StageEvaluation,
) -> list[dict[str, object]]:
    final_scores = aggregate_feature_matrices(feature_matrices, final_stage.feature_names, final_stage.weights)
    rows: list[dict[str, object]] = []
    for source_index, source in enumerate(templates):
        row_scores = final_scores[source_index]
        ranking = np.argsort(-row_scores)
        for rank, target_index in enumerate(ranking, start=1):
            target = templates[int(target_index)]
            rows.append(
                {
                    "SourceIndex": int(source.character_index),
                    "SourceName": str(source.character_name),
                    "SourceRosterIndex": int(source.roster_index),
                    "TargetIndex": int(target.character_index),
                    "TargetName": str(target.character_name),
                    "TargetRosterIndex": int(target.roster_index),
                    "RankByFinalScore": int(rank),
                    "IsSelf": int(source.character_index == target.character_index),
                    "ColorScore": round(float(feature_matrices["color"][source_index, target_index]) * 100.0, 3),
                    "GrayScore": round(float(feature_matrices["gray"][source_index, target_index]) * 100.0, 3),
                    "EdgeScore": round(float(feature_matrices["edge"][source_index, target_index]) * 100.0, 3),
                    "CutoutColorScore": round(float(feature_matrices["cutout_color"][source_index, target_index]) * 100.0, 3),
                    "CutoutGrayScore": round(float(feature_matrices["cutout_gray"][source_index, target_index]) * 100.0, 3),
                    "CutoutEdgeScore": round(float(feature_matrices["cutout_edge"][source_index, target_index]) * 100.0, 3),
                    "SilhouetteScore": round(float(feature_matrices["silhouette"][source_index, target_index]) * 100.0, 3),
                    "ContourScore": round(float(feature_matrices["contour"][source_index, target_index]) * 100.0, 3),
                    "FinalScore": round(float(final_scores[source_index, target_index]) * 100.0, 3),
                }
            )
    return rows


def build_stage_summary_rows(stages: Sequence[StageEvaluation]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for stage in stages:
        row = {
            "StageName": str(stage.stage_name),
            "Features": ",".join(stage.feature_names),
            "Weights": ",".join(f"{weight:.2f}" for weight in stage.weights),
            "CorrectSelfMatches": int(stage.correct_count),
            "TemplateCount": int(stage.total_count),
            "SelfMatchAccuracy": round(100.0 * stage.correct_count / max(1, stage.total_count), 3),
            "MinSelfMargin": round(float(stage.min_margin) * 100.0, 3),
            "MedianSelfMargin": round(float(stage.median_margin) * 100.0, 3),
            "MeanSelfMargin": round(float(stage.mean_margin) * 100.0, 3),
        }
        rows.append(row)
    return rows


def build_nearest_neighbor_rows(
    templates: Sequence[TemplateRecord],
    feature_matrices: dict[str, np.ndarray],
    final_stage: StageEvaluation,
) -> list[dict[str, object]]:
    final_scores = aggregate_feature_matrices(feature_matrices, final_stage.feature_names, final_stage.weights)
    rows: list[dict[str, object]] = []
    for source_index, source in enumerate(templates):
        row_scores = final_scores[source_index].copy()
        self_score = float(row_scores[source_index])
        row_scores[source_index] = -np.inf
        best_other_index = int(np.argmax(row_scores))
        best_other_score = float(row_scores[best_other_index])
        target = templates[best_other_index]
        rows.append(
            {
                "SourceIndex": int(source.character_index),
                "SourceName": str(source.character_name),
                "SourceRosterIndex": int(source.roster_index),
                "SelfScore": round(self_score * 100.0, 3),
                "NearestIndex": int(target.character_index),
                "NearestName": str(target.character_name),
                "NearestRosterIndex": int(target.roster_index),
                "NearestScore": round(best_other_score * 100.0, 3),
                "Margin": round((self_score - best_other_score) * 100.0, 3),
                "NearestColorScore": round(float(feature_matrices["color"][source_index, best_other_index]) * 100.0, 3),
                "NearestGrayScore": round(float(feature_matrices["gray"][source_index, best_other_index]) * 100.0, 3),
                "NearestEdgeScore": round(float(feature_matrices["edge"][source_index, best_other_index]) * 100.0, 3),
                "NearestCutoutColorScore": round(float(feature_matrices["cutout_color"][source_index, best_other_index]) * 100.0, 3),
                "NearestCutoutGrayScore": round(float(feature_matrices["cutout_gray"][source_index, best_other_index]) * 100.0, 3),
                "NearestCutoutEdgeScore": round(float(feature_matrices["cutout_edge"][source_index, best_other_index]) * 100.0, 3),
                "NearestSilhouetteScore": round(float(feature_matrices["silhouette"][source_index, best_other_index]) * 100.0, 3),
                "NearestContourScore": round(float(feature_matrices["contour"][source_index, best_other_index]) * 100.0, 3),
            }
        )
    rows.sort(key=lambda item: float(item["Margin"]))
    return rows


def build_hard_pair_rows(
    templates: Sequence[TemplateRecord],
    feature_matrices: dict[str, np.ndarray],
    final_stage: StageEvaluation,
    *,
    limit: int = 40,
) -> list[dict[str, object]]:
    final_scores = aggregate_feature_matrices(feature_matrices, final_stage.feature_names, final_stage.weights)
    rows: list[dict[str, object]] = []
    for left_index in range(len(templates)):
        for right_index in range(left_index + 1, len(templates)):
            left = templates[left_index]
            right = templates[right_index]
            left_to_right = float(final_scores[left_index, right_index])
            right_to_left = float(final_scores[right_index, left_index])
            symmetric_score = (left_to_right + right_to_left) / 2.0
            rows.append(
                {
                    "LeftIndex": int(left.character_index),
                    "LeftName": str(left.character_name),
                    "LeftRosterIndex": int(left.roster_index),
                    "RightIndex": int(right.character_index),
                    "RightName": str(right.character_name),
                    "RightRosterIndex": int(right.roster_index),
                    "SymmetricFinalScore": round(symmetric_score * 100.0, 3),
                    "LeftToRightScore": round(left_to_right * 100.0, 3),
                    "RightToLeftScore": round(right_to_left * 100.0, 3),
                    "ColorScoreAvg": round(((float(feature_matrices["color"][left_index, right_index]) + float(feature_matrices["color"][right_index, left_index])) / 2.0) * 100.0, 3),
                    "GrayScoreAvg": round(((float(feature_matrices["gray"][left_index, right_index]) + float(feature_matrices["gray"][right_index, left_index])) / 2.0) * 100.0, 3),
                    "EdgeScoreAvg": round(((float(feature_matrices["edge"][left_index, right_index]) + float(feature_matrices["edge"][right_index, left_index])) / 2.0) * 100.0, 3),
                    "CutoutColorScoreAvg": round(((float(feature_matrices["cutout_color"][left_index, right_index]) + float(feature_matrices["cutout_color"][right_index, left_index])) / 2.0) * 100.0, 3),
                    "CutoutGrayScoreAvg": round(((float(feature_matrices["cutout_gray"][left_index, right_index]) + float(feature_matrices["cutout_gray"][right_index, left_index])) / 2.0) * 100.0, 3),
                    "CutoutEdgeScoreAvg": round(((float(feature_matrices["cutout_edge"][left_index, right_index]) + float(feature_matrices["cutout_edge"][right_index, left_index])) / 2.0) * 100.0, 3),
                    "SilhouetteScoreAvg": round(((float(feature_matrices["silhouette"][left_index, right_index]) + float(feature_matrices["silhouette"][right_index, left_index])) / 2.0) * 100.0, 3),
                    "ContourScoreAvg": round(((float(feature_matrices["contour"][left_index, right_index]) + float(feature_matrices["contour"][right_index, left_index])) / 2.0) * 100.0, 3),
                }
            )
    rows.sort(key=lambda item: float(item["SymmetricFinalScore"]), reverse=True)
    return rows[: int(limit)]


def write_markdown_summary(path: Path, stages: Sequence[StageEvaluation], nearest_rows: Sequence[dict[str, object]], hard_pair_rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    final_stage = stages[-1]
    hardest_template_rows = nearest_rows[:10]
    hardest_pairs = hard_pair_rows[:10]

    lines = [
        "# Character Roster Template Evaluation",
        "",
        "## Final Generic Scorer",
        f"- Features: `{','.join(final_stage.feature_names)}`",
        f"- Weights: `{','.join(f'{weight:.2f}' for weight in final_stage.weights)}`",
        f"- Self matches correct: `{final_stage.correct_count}/{final_stage.total_count}`",
        f"- Min self margin: `{final_stage.min_margin * 100.0:.3f}`",
        f"- Median self margin: `{final_stage.median_margin * 100.0:.3f}`",
        f"- Mean self margin: `{final_stage.mean_margin * 100.0:.3f}`",
        "",
        "## Stage Progression",
    ]
    for stage in stages:
        lines.append(
            f"- `{stage.stage_name}` | weights `{','.join(f'{weight:.2f}' for weight in stage.weights)}` | correct `{stage.correct_count}/{stage.total_count}` | min margin `{stage.min_margin * 100.0:.3f}`"
        )

    lines.extend(["", "## Hardest Templates", "| Source | Nearest Nonself | Margin |", "| --- | --- | ---: |"])
    for row in hardest_template_rows:
        lines.append(f"| {row['SourceName']} ({row['SourceIndex']}) | {row['NearestName']} ({row['NearestIndex']}) | {row['Margin']:.3f} |")

    lines.extend(["", "## Hardest Pairs", "| Left | Right | Symmetric Score |", "| --- | --- | ---: |"])
    for row in hardest_pairs:
        lines.append(f"| {row['LeftName']} ({row['LeftIndex']}) | {row['RightName']} ({row['RightIndex']}) | {row['SymmetricFinalScore']:.3f} |")

    path.write_text("\n".join(lines), encoding="utf-8")


def _rank_crop_against_templates(
    crop_image: np.ndarray,
    templates: Sequence[TemplateRecord],
) -> list[dict[str, object]]:
    if not templates:
        return []
    target_height, target_width = templates[0].template_image.shape[:2]
    if crop_image.shape[0] != target_height or crop_image.shape[1] != target_width:
        crop_image = cv2.resize(crop_image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    crop_alpha = np.full(crop_image.shape[:2], 255, dtype=np.uint8)
    ranked_rows: list[dict[str, object]] = []
    for template in templates:
        color_score = masked_color_score(crop_image, crop_alpha, template.template_image, template.template_alpha)
        gray_score = masked_grayscale_score(crop_image, crop_alpha, template.template_image, template.template_alpha)
        edge_score = edge_agreement_score(crop_image, crop_alpha, template.template_image, template.template_alpha)
        cutout_color_score = masked_cutout_color_score(crop_image, crop_alpha, template.template_image, template.template_alpha)
        cutout_gray_score = masked_cutout_grayscale_score(crop_image, crop_alpha, template.template_image, template.template_alpha)
        cutout_edge_score = cutout_edge_agreement_score(crop_image, crop_alpha, template.template_image, template.template_alpha)
        aligned_blend_score, aligned_color_score, aligned_edge_score, aligned_dx, aligned_dy = aligned_cutout_blend_score(
            crop_image,
            crop_alpha,
            template.template_image,
            template.template_alpha,
        )
        runtime_blend = (0.20 * color_score) + (0.80 * edge_score)
        cutout_blend = (0.20 * cutout_color_score) + (0.80 * cutout_edge_score)
        ranked_rows.append(
            {
                "TargetIndex": int(template.character_index),
                "TargetName": str(template.character_name),
                "TargetRosterIndex": int(template.roster_index),
                "RuntimeBlendScore": float(runtime_blend),
                "ColorScore": float(color_score),
                "GrayScore": float(gray_score),
                "EdgeScore": float(edge_score),
                "CutoutBlendScore": float(cutout_blend),
                "CutoutColorScore": float(cutout_color_score),
                "CutoutGrayScore": float(cutout_gray_score),
                "CutoutEdgeScore": float(cutout_edge_score),
                "AlignedCutoutBlendScore": float(aligned_blend_score),
                "AlignedCutoutColorScore": float(aligned_color_score),
                "AlignedCutoutEdgeScore": float(aligned_edge_score),
                "AlignedDx": int(aligned_dx),
                "AlignedDy": int(aligned_dy),
            }
        )
    runtime_order = sorted(range(len(ranked_rows)), key=lambda index: ranked_rows[index]["RuntimeBlendScore"], reverse=True)
    cutout_order = sorted(range(len(ranked_rows)), key=lambda index: ranked_rows[index]["CutoutBlendScore"], reverse=True)
    aligned_cutout_order = sorted(range(len(ranked_rows)), key=lambda index: ranked_rows[index]["AlignedCutoutBlendScore"], reverse=True)
    for rank, index in enumerate(runtime_order, start=1):
        ranked_rows[index]["RuntimeRank"] = int(rank)
    for rank, index in enumerate(cutout_order, start=1):
        ranked_rows[index]["CutoutRank"] = int(rank)
    for rank, index in enumerate(aligned_cutout_order, start=1):
        ranked_rows[index]["AlignedCutoutRank"] = int(rank)
    ranked_rows.sort(key=lambda item: int(item["RuntimeRank"]))
    return ranked_rows


def run_crop_probe(
    *,
    templates: Sequence[TemplateRecord],
    debug_csv: Path,
    output_dir: Path,
    cases: Sequence[str],
    bundles: Sequence[str],
    topn: int = 5,
) -> Path:
    requested_cases = {item.strip() for item in cases if item.strip()}
    rows: list[dict[str, object]] = []
    with debug_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        debug_rows = list(csv.DictReader(handle))

    for debug_row in debug_rows:
        video = str(debug_row.get("Video") or "")
        player = str(debug_row.get("Standardized Player") or "")
        if requested_cases and f"{video}|{player}" not in requested_cases:
            continue
        race = int(debug_row.get("Race", 0) or 0)
        position = int(debug_row.get("Position", 0) or 0)
        if race <= 0 or position <= 0:
            continue
        for bundle in bundles:
            frame_path = find_score_bundle_anchor_path(video, race, bundle)
            if frame_path is None or not Path(frame_path).exists():
                continue
            frame_image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame_image is None or frame_image.size == 0:
                continue
            score_layout_id = score_layout_id_from_filename(frame_path)
            (x1, y1), (x2, y2) = character_row_roi(position - 1, score_layout_id=score_layout_id)
            crop_image = frame_image[y1:y2, x1:x2]
            if crop_image.size == 0:
                continue
            all_ranked = _rank_crop_against_templates(crop_image, templates)
            top_limit = max(1, int(topn))
            ranked = [
                item
                for item in all_ranked
                if int(item["RuntimeRank"]) <= top_limit
                or int(item["CutoutRank"]) <= top_limit
                or int(item["AlignedCutoutRank"]) <= top_limit
            ]
            ranked.sort(key=lambda item: (int(item["RuntimeRank"]), int(item["CutoutRank"]), int(item["AlignedCutoutRank"])))
            for rank, item in enumerate(ranked, start=1):
                rows.append(
                    {
                        "Video": video,
                        "Player": player,
                        "Race": race,
                        "Position": position,
                        "Bundle": bundle,
                        "Rank": rank,
                        "RuntimeRank": int(item["RuntimeRank"]),
                        "CutoutRank": int(item["CutoutRank"]),
                        "AlignedCutoutRank": int(item["AlignedCutoutRank"]),
                        "TargetIndex": item["TargetIndex"],
                        "TargetName": item["TargetName"],
                        "TargetRosterIndex": item["TargetRosterIndex"],
                        "RuntimeBlendScore": round(item["RuntimeBlendScore"] * 100.0, 3),
                        "ColorScore": round(item["ColorScore"] * 100.0, 3),
                        "GrayScore": round(item["GrayScore"] * 100.0, 3),
                        "EdgeScore": round(item["EdgeScore"] * 100.0, 3),
                        "CutoutBlendScore": round(item["CutoutBlendScore"] * 100.0, 3),
                        "CutoutColorScore": round(item["CutoutColorScore"] * 100.0, 3),
                        "CutoutGrayScore": round(item["CutoutGrayScore"] * 100.0, 3),
                        "CutoutEdgeScore": round(item["CutoutEdgeScore"] * 100.0, 3),
                        "AlignedCutoutBlendScore": round(item["AlignedCutoutBlendScore"] * 100.0, 3),
                        "AlignedCutoutColorScore": round(item["AlignedCutoutColorScore"] * 100.0, 3),
                        "AlignedCutoutEdgeScore": round(item["AlignedCutoutEdgeScore"] * 100.0, 3),
                        "AlignedDx": int(item["AlignedDx"]),
                        "AlignedDy": int(item["AlignedDy"]),
                    }
                )

    if not rows:
        raise RuntimeError("No crop probe rows were generated.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "crop_probe.csv"
    write_csv(output_path, rows[0].keys(), rows)
    return output_path


def _mode_offset(rows: Sequence[dict[str, object]]) -> tuple[int, int]:
    counter = Counter((int(row["TruthAlignedDx"]), int(row["TruthAlignedDy"])) for row in rows)
    if not counter:
        return 0, 0
    best_count = max(counter.values())
    candidates = [offset for offset, count in counter.items() if count == best_count]
    if len(candidates) == 1:
        return candidates[0]
    avg_score_by_offset: dict[tuple[int, int], float] = {}
    for offset in candidates:
        offset_rows = [
            row for row in rows
            if (int(row["TruthAlignedDx"]), int(row["TruthAlignedDy"])) == offset
        ]
        avg_score_by_offset[offset] = float(np.mean([float(row["TruthAlignedCutoutBlendScore"]) for row in offset_rows]))
    return max(candidates, key=lambda offset: (avg_score_by_offset[offset], -abs(offset[0]), -abs(offset[1])))


def _row_as_float(row: dict[str, object], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _truth_offset_probe(
    crop_image: np.ndarray,
    template: TemplateRecord,
) -> dict[str, object]:
    target_height, target_width = template.template_image.shape[:2]
    if crop_image.shape[0] != target_height or crop_image.shape[1] != target_width:
        crop_image = cv2.resize(crop_image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    crop_alpha = np.full(crop_image.shape[:2], 255, dtype=np.uint8)
    runtime_color = masked_color_score(crop_image, crop_alpha, template.template_image, template.template_alpha)
    runtime_edge = edge_agreement_score(crop_image, crop_alpha, template.template_image, template.template_alpha)
    cutout_color = masked_cutout_color_score(crop_image, crop_alpha, template.template_image, template.template_alpha)
    cutout_edge = cutout_edge_agreement_score(crop_image, crop_alpha, template.template_image, template.template_alpha)
    aligned_blend, aligned_color, aligned_edge, aligned_dx, aligned_dy = aligned_cutout_blend_score(
        crop_image,
        crop_alpha,
        template.template_image,
        template.template_alpha,
    )
    return {
        "RuntimeBlendScore": (0.20 * runtime_color) + (0.80 * runtime_edge),
        "CutoutBlendScore": (0.20 * cutout_color) + (0.80 * cutout_edge),
        "AlignedCutoutBlendScore": aligned_blend,
        "AlignedCutoutColorScore": aligned_color,
        "AlignedCutoutEdgeScore": aligned_edge,
        "AlignedDx": aligned_dx,
        "AlignedDy": aligned_dy,
    }


def _frame_quality_metrics(frame_image: np.ndarray) -> dict[str, object]:
    gray = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)

    def black_border_thickness(lines: np.ndarray) -> int:
        thickness = 0
        for line in lines:
            if float(np.mean(line < 8)) >= 0.98:
                thickness += 1
            else:
                break
        return int(thickness)

    return {
        "TopBorder": black_border_thickness(gray),
        "BottomBorder": black_border_thickness(gray[::-1]),
        "LeftBorder": black_border_thickness(gray.T),
        "RightBorder": black_border_thickness(gray.T[::-1]),
        "Sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 3),
    }


def _write_character_offset_recommendations(output_dir: Path, rows: Sequence[dict[str, object]]) -> Path:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["CharacterIndex"])].append(row)

    recommendation_rows: list[dict[str, object]] = []
    for character_index, character_rows in sorted(grouped.items()):
        recommended_dx, recommended_dy = _mode_offset(character_rows)
        matching_offset_rows = [
            row for row in character_rows
            if int(row["TruthAlignedDx"]) == recommended_dx and int(row["TruthAlignedDy"]) == recommended_dy
        ]
        recommendation_rows.append(
            {
                "CharacterIndex": int(character_index),
                "Character": str(character_rows[0]["Character"]),
                "Roster": str(character_rows[0]["Roster"]),
                "RecommendedDx": int(recommended_dx),
                "RecommendedDy": int(recommended_dy),
                "Observations": int(len(character_rows)),
                "RecommendedOffsetCount": int(len(matching_offset_rows)),
                "RecommendedOffsetShare": round(len(matching_offset_rows) / max(1, len(character_rows)), 3),
                "AvgTruthAlignedScore": round(float(np.mean([_row_as_float(row, "TruthAlignedCutoutBlendScore") for row in character_rows])), 3),
                "MinTruthAlignedScore": round(float(np.min([_row_as_float(row, "TruthAlignedCutoutBlendScore") for row in character_rows])), 3),
                "AvgSharpness": round(float(np.mean([_row_as_float(row, "Sharpness") for row in character_rows])), 3),
                "MaxBlackBorder": int(
                    max(
                        max(
                            int(row.get("TopBorder", 0) or 0),
                            int(row.get("BottomBorder", 0) or 0),
                            int(row.get("LeftBorder", 0) or 0),
                            int(row.get("RightBorder", 0) or 0),
                        )
                        for row in character_rows
                    )
                ),
                "OffsetsSeen": " ".join(
                    f"{dx},{dy}:{count}"
                    for (dx, dy), count in sorted(
                        Counter((int(row["TruthAlignedDx"]), int(row["TruthAlignedDy"])) for row in character_rows).items(),
                        key=lambda item: (-item[1], item[0][0], item[0][1]),
                    )
                ),
            }
        )

    output_path = output_dir / "character_offset_recommendations.csv"
    write_csv(output_path, recommendation_rows[0].keys(), recommendation_rows)
    return output_path


def run_offset_calibration(
    *,
    templates: Sequence[TemplateRecord],
    targets_csv: Path,
    output_dir: Path,
    min_width: int = 1280,
    min_height: int = 720,
) -> tuple[Path, Path]:
    templates_by_index = {int(template.character_index): template for template in templates}
    rows: list[dict[str, object]] = []
    with targets_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        target_rows = list(csv.DictReader(handle))

    for target_row in target_rows:
        character_index = int(target_row["CharacterIndex"])
        target_template = templates_by_index.get(character_index)
        if target_template is None:
            continue
        position = int(target_row["Position"])
        for bundle, path_key in (("2RaceScore", "RaceScoreAnchor"), ("3TotalScore", "TotalScoreAnchor")):
            frame_path = Path(str(target_row.get(path_key) or ""))
            if not frame_path.exists():
                continue
            frame_image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame_image is None or frame_image.size == 0:
                continue
            frame_height, frame_width = frame_image.shape[:2]
            if frame_width < int(min_width) or frame_height < int(min_height):
                continue
            quality = _frame_quality_metrics(frame_image)
            score_layout_id = score_layout_id_from_filename(frame_path)
            (x1, y1), (x2, y2) = character_row_roi(position - 1, score_layout_id=score_layout_id)
            crop_image = frame_image[y1:y2, x1:x2]
            if crop_image.size == 0:
                continue

            truth_rank = _truth_offset_probe(crop_image, target_template)
            rows.append(
                {
                    "CharacterIndex": character_index,
                    "Character": str(target_row["Character"]),
                    "Roster": str(target_row["Roster"]),
                    "Video": str(target_row["Video"]),
                    "Race": int(target_row["Race"]),
                    "Position": position,
                    "Bundle": bundle,
                    "FrameWidth": int(frame_width),
                    "FrameHeight": int(frame_height),
                    **quality,
                    "TruthRuntimeBlendScore": round(float(truth_rank["RuntimeBlendScore"]) * 100.0, 3),
                    "TruthCutoutBlendScore": round(float(truth_rank["CutoutBlendScore"]) * 100.0, 3),
                    "TruthAlignedCutoutBlendScore": round(float(truth_rank["AlignedCutoutBlendScore"]) * 100.0, 3),
                    "TruthAlignedColorScore": round(float(truth_rank["AlignedCutoutColorScore"]) * 100.0, 3),
                    "TruthAlignedEdgeScore": round(float(truth_rank["AlignedCutoutEdgeScore"]) * 100.0, 3),
                    "TruthAlignedDx": int(truth_rank["AlignedDx"]),
                    "TruthAlignedDy": int(truth_rank["AlignedDy"]),
                    "FramePath": str(frame_path),
                }
            )

    if not rows:
        raise RuntimeError("No offset calibration rows were generated.")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_output_path = output_dir / "character_offset_calibration_probe.csv"
    write_csv(raw_output_path, rows[0].keys(), rows)
    recommendations_path = _write_character_offset_recommendations(output_dir, rows)
    return raw_output_path, recommendations_path


def run_evaluation(
    *,
    asset_dir: Path,
    output_dir: Path,
    start_index: int = 0,
    end_index: int = 78,
    template_size: int = CHARACTER_TEMPLATE_SIZE,
    weight_step: float = 0.05,
    hard_pair_limit: int = 40,
) -> dict[str, object]:
    templates = load_runtime_roster_templates(
        asset_dir=asset_dir,
        start_index=start_index,
        end_index=end_index,
        template_size=template_size,
    )
    feature_matrices = compute_feature_matrices(templates)
    stages = build_stage_sequence(feature_matrices, weight_step=weight_step)
    final_stage = stages[-1]
    pairwise_rows = build_pairwise_rows(templates, feature_matrices, final_stage)
    stage_summary_rows = build_stage_summary_rows(stages)
    nearest_rows = build_nearest_neighbor_rows(templates, feature_matrices, final_stage)
    hard_pair_rows = build_hard_pair_rows(templates, feature_matrices, final_stage, limit=hard_pair_limit)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "stage_summary.csv",
        stage_summary_rows[0].keys(),
        stage_summary_rows,
    )
    write_csv(
        output_dir / "nearest_neighbors.csv",
        nearest_rows[0].keys(),
        nearest_rows,
    )
    write_csv(
        output_dir / "hard_pairs.csv",
        hard_pair_rows[0].keys(),
        hard_pair_rows,
    )
    write_csv(
        output_dir / "pairwise_scores.csv",
        pairwise_rows[0].keys(),
        pairwise_rows,
    )
    write_markdown_summary(output_dir / "summary.md", stages, nearest_rows, hard_pair_rows)

    return {
        "templates": templates,
        "feature_matrices": feature_matrices,
        "stages": stages,
        "final_stage": final_stage,
        "nearest_rows": nearest_rows,
        "hard_pair_rows": hard_pair_rows,
        "output_dir": output_dir,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate how strongly runtime-sized roster character templates separate from each other.")
    parser.add_argument("--asset-dir", default="assets/character", help="Directory containing character template PNG files.")
    parser.add_argument("--output-dir", default="Output_Results/Debug/character_roster_template_eval", help="Directory to write CSV and Markdown reports.")
    parser.add_argument("--start-index", type=int, default=0, help="First roster template index to include.")
    parser.add_argument("--end-index", type=int, default=78, help="Last roster template index to include.")
    parser.add_argument("--template-size", type=int, default=CHARACTER_TEMPLATE_SIZE, help="Normalized runtime template size.")
    parser.add_argument("--weight-step", type=float, default=0.05, help="Search step for feature weights. Smaller values take longer.")
    parser.add_argument("--hard-pair-limit", type=int, default=40, help="How many highest-scoring nonself pairs to keep in the report.")
    parser.add_argument("--probe-debug-csv", help="Optional debug CSV to probe saved race/total character crops against the roster templates.")
    parser.add_argument("--probe-case", action="append", default=[], help="Video|Player probe case to evaluate from the debug CSV. Can be passed multiple times.")
    parser.add_argument("--probe-bundles", nargs="+", default=("2RaceScore", "3TotalScore"), help="Saved score bundles to probe when --probe-debug-csv is provided.")
    parser.add_argument("--probe-topn", type=int, default=5, help="Number of ranked template matches to emit per probed crop.")
    parser.add_argument("--offset-calibration-targets", help="Optional target CSV with known CharacterIndex, Position, and saved score anchors.")
    parser.add_argument("--offset-min-width", type=int, default=1280, help="Minimum frame width for offset calibration targets.")
    parser.add_argument("--offset-min-height", type=int, default=720, help="Minimum frame height for offset calibration targets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_evaluation(
        asset_dir=Path(args.asset_dir),
        output_dir=Path(args.output_dir),
        start_index=int(args.start_index),
        end_index=int(args.end_index),
        template_size=int(args.template_size),
        weight_step=float(args.weight_step),
        hard_pair_limit=int(args.hard_pair_limit),
    )
    final_stage: StageEvaluation = results["final_stage"]
    output_dir: Path = results["output_dir"]
    if args.probe_debug_csv:
        probe_output = run_crop_probe(
            templates=results["templates"],
            debug_csv=Path(args.probe_debug_csv),
            output_dir=output_dir,
            cases=list(args.probe_case),
            bundles=list(args.probe_bundles),
            topn=int(args.probe_topn),
        )
        print(f"crop probe: {probe_output}")
    if args.offset_calibration_targets:
        raw_output, recommendations_output = run_offset_calibration(
            templates=results["templates"],
            targets_csv=Path(args.offset_calibration_targets),
            output_dir=output_dir,
            min_width=int(args.offset_min_width),
            min_height=int(args.offset_min_height),
        )
        print(f"offset calibration: {raw_output}")
        print(f"offset recommendations: {recommendations_output}")
    print(
        f"{output_dir} | final stage {final_stage.stage_name} | "
        f"correct {final_stage.correct_count}/{final_stage.total_count} | "
        f"min margin {final_stage.min_margin * 100.0:.3f}"
    )


if __name__ == "__main__":
    main()
