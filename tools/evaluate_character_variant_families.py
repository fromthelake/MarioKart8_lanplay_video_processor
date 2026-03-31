import argparse
import csv
from pathlib import Path

import cv2

from mk8_local_play.extract_text import (
    _masked_chroma_variant_score,
    build_character_variant_family_diagnostic_mask,
    character_variant_family_templates,
    diagnostic_character_variant_score,
    resolve_character_variant_family_name,
)
from mk8_local_play.ocr_scoreboard_consensus import best_character_matches, load_character_templates


def _read_probe_metadata(candidate_report_path: Path) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    if not candidate_report_path.exists():
        return metadata

    with candidate_report_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            crop_file = str(row.get("crop_file") or "").strip()
            if not crop_file or crop_file in metadata:
                continue
            metadata[crop_file] = row
    return metadata


def _rank_family_chroma(roi, family_templates):
    ranked = []
    for template in family_templates:
        score = _masked_chroma_variant_score(
            roi,
            template["template_image"],
            template["template_alpha"],
        )
        ranked.append(
            {
                "Character": str(template["character_name"]),
                "CharacterIndex": int(template["character_index"]),
                "CharacterMatchConfidence": round(float(score) * 100.0, 1),
                "CharacterMatchMethod": "family_chroma_variant_probe",
            }
        )
    ranked.sort(key=lambda item: item["CharacterMatchConfidence"], reverse=True)
    return ranked


def _rank_family_diagnostic(roi, family_templates):
    diagnostic_mask = build_character_variant_family_diagnostic_mask(family_templates)
    ranked = []
    for template in family_templates:
        score = diagnostic_character_variant_score(
            roi,
            template["template_image"],
            diagnostic_mask,
        )
        ranked.append(
            {
                "Character": str(template["character_name"]),
                "CharacterIndex": int(template["character_index"]),
                "CharacterMatchConfidence": round(float(score) * 100.0, 1),
                "CharacterMatchMethod": "family_diagnostic_variant_probe",
            }
        )
    ranked.sort(key=lambda item: item["CharacterMatchConfidence"], reverse=True)
    return ranked


def main():
    parser = argparse.ArgumentParser(description="Evaluate variant-family character rankings on saved ROI crops.")
    parser.add_argument("--crop-dir", required=True, help="Directory containing saved character ROI PNG crops.")
    parser.add_argument(
        "--candidate-report",
        default="candidate_report.csv",
        help="Optional probe report CSV in the crop directory used to recover original context.",
    )
    parser.add_argument(
        "--output",
        default="family_variant_report.csv",
        help="Output CSV filename written inside the crop directory by default.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of ranked matches to write per method.",
    )
    args = parser.parse_args()

    crop_dir = Path(args.crop_dir)
    candidate_report_path = crop_dir / args.candidate_report
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = crop_dir / output_path

    templates = load_character_templates()
    metadata_by_crop = _read_probe_metadata(candidate_report_path)
    rows = []

    for crop_path in sorted(crop_dir.glob("*.png")):
        roi = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        if roi is None or roi.size == 0:
            continue

        full_matches = best_character_matches(roi, templates, limit=max(2, int(args.limit)))
        if not full_matches:
            continue

        top_character = str(full_matches[0].get("Character") or "")
        family_name = resolve_character_variant_family_name(top_character)
        family_templates = character_variant_family_templates(templates, top_character)
        if not family_name or not family_templates:
            continue

        family_matches = best_character_matches(roi, family_templates, limit=max(2, int(args.limit)))
        chroma_matches = _rank_family_chroma(roi, family_templates)[: max(2, int(args.limit))]
        diagnostic_matches = _rank_family_diagnostic(roi, family_templates)[: max(2, int(args.limit))]
        probe_metadata = metadata_by_crop.get(crop_path.name, {})

        for method_name, matches in (
            ("full_search", full_matches[: int(args.limit)]),
            ("family_search", family_matches[: int(args.limit)]),
            ("family_chroma", chroma_matches[: int(args.limit)]),
            ("family_diagnostic", diagnostic_matches[: int(args.limit)]),
        ):
            for rank, match in enumerate(matches, start=1):
                rows.append(
                    {
                        "crop_file": crop_path.name,
                        "family_name": family_name,
                        "probe_video": str(probe_metadata.get("video") or ""),
                        "probe_player": str(probe_metadata.get("player") or ""),
                        "probe_race": str(probe_metadata.get("race") or ""),
                        "probe_position": str(probe_metadata.get("position") or ""),
                        "probe_export_character_before_mii": str(probe_metadata.get("export_character_before_mii") or ""),
                        "probe_raw_best": str(probe_metadata.get("raw_best") or ""),
                        "probe_raw_margin": str(probe_metadata.get("raw_margin") or ""),
                        "method": method_name,
                        "rank": rank,
                        "candidate_character": str(match["Character"]),
                        "candidate_index": int(match["CharacterIndex"]),
                        "candidate_confidence": float(match["CharacterMatchConfidence"]),
                        "candidate_method": str(match["CharacterMatchMethod"]),
                    }
                )

    if not rows:
        raise RuntimeError(f"No variant-family crops were evaluated in {crop_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(output_path)


if __name__ == "__main__":
    main()
