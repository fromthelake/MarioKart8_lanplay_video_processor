from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from mk8_local_play.data_paths import resolve_asset_file
from mk8_local_play.extract_common import TARGET_HEIGHT, TARGET_WIDTH, crop_and_upscale_image
from mk8_local_play.extract_initial_scan import INITIAL_SCAN_TARGETS


TEMPLATE_GRID_X = 313
TEMPLATE_GRID_Y = 46
TEMPLATE_GRID_SIZE = 52


def load_alpha_tile(pos_number: int, template_variant: str) -> tuple[object, object | None]:
    template_name = f"Score_template_{template_variant}.png"
    src = resolve_asset_file("templates", template_name)
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load template: {src}")
    y = (int(pos_number) - 1) * 52
    tile = img[y:y + 52, 0:52]
    if tile.shape[0] != 52 or tile.shape[1] != 52:
        raise ValueError(f"Unexpected tile size for {template_name} pos {pos_number}: {tile.shape}")
    if len(tile.shape) == 3 and tile.shape[2] == 4:
        bgr = tile[:, :, :3]
        alpha = tile[:, :, 3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, alpha_mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
    elif len(tile.shape) == 3:
        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        alpha_mask = None
    else:
        gray = tile
        alpha_mask = None
    return gray, alpha_mask


def match_masked_template(search_gray, template_gray, alpha_mask):
    if search_gray.shape[0] < template_gray.shape[0] or search_gray.shape[1] < template_gray.shape[1]:
        search_gray = cv2.resize(
            search_gray,
            (max(template_gray.shape[1], search_gray.shape[1]), max(template_gray.shape[0], search_gray.shape[0])),
            interpolation=cv2.INTER_LINEAR,
        )
    result = cv2.matchTemplate(search_gray, template_gray, cv2.TM_CCOEFF_NORMED, mask=alpha_mask)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return float(max_val), tuple(int(v) for v in max_loc)


def row_search_roi_from_upscaled(upscaled_bgr, row_number: int):
    x1 = int(TEMPLATE_GRID_X)
    y1 = int(TEMPLATE_GRID_Y) + ((int(row_number) - 1) * int(TEMPLATE_GRID_SIZE))
    x2 = x1 + int(TEMPLATE_GRID_SIZE)
    y2 = y1 + int(TEMPLATE_GRID_SIZE)
    return upscaled_bgr[y1:y2, x1:x2]


def _best_inverse_match(search_gray, pos_number: int) -> dict:
    results = []
    threshold = 0.40
    for variant in ("white", "black"):
        tile_gray, alpha_mask = load_alpha_tile(pos_number, variant)
        score, loc = match_masked_template(search_gray, tile_gray, alpha_mask)
        result = {
            "variant": variant,
            "score": float(score),
            "loc": tuple(int(v) for v in loc),
        }
        results.append(result)
        if float(score) >= threshold:
            best = result
            return {
                "best_variant": str(best["variant"]),
                "best_score": float(best["score"]),
                "best_loc": tuple(int(v) for v in best["loc"]),
                "variants": results,
            }
    best = max(results, key=lambda item: item["score"])
    return {
        "best_variant": str(best["variant"]),
        "best_score": float(best["score"]),
        "best_loc": tuple(int(v) for v in best["loc"]),
        "variants": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe raw score tile matching against an unprocessed upscaled frame.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame", required=True, type=int)
    parser.add_argument("--positions", nargs="*", type=int, default=[5, 6])
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--dump-dir")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        alt = Path("Input_Videos") / args.video
        if alt.exists():
            video_path = alt

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.frame))
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Could not read frame {args.frame}")
    actual_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    cap.release()

    upscaled = crop_and_upscale_image(frame, 0, 0, frame.shape[1], frame.shape[0], TARGET_WIDTH, TARGET_HEIGHT)
    score_target = next(target for target in INITIAL_SCAN_TARGETS if target["kind"] == "score")
    x, y, w, h = score_target["roi"]
    score_roi = upscaled[y:y + h, x:x + w]
    score_roi_gray = cv2.cvtColor(score_roi, cv2.COLOR_BGR2GRAY)

    dump_dir = Path(args.dump_dir) if args.dump_dir else Path("Output_Results") / "Debug" / f"{video_path.stem}_raw_tile_probe_f{actual_frame}"
    dump_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dump_dir / "frame_upscaled.jpg"), upscaled)
    cv2.imwrite(str(dump_dir / "score_roi.png"), score_roi)

    print(f"video={video_path.name}")
    print(f"requested_frame={args.frame}")
    print(f"actual_frame={actual_frame}")
    print(f"threshold={float(args.threshold):.3f}")
    print(f"dump_dir={dump_dir}")
    print("whole score ROI matches (best of black/white):")
    whole_scores = {}
    for pos in args.positions:
        match_info = _best_inverse_match(score_roi_gray, pos)
        whole_scores[int(pos)] = float(match_info["best_score"])
        for variant in ("black", "white"):
            tile_gray, _alpha_mask = load_alpha_tile(pos, variant)
            cv2.imwrite(str(dump_dir / f"template_pos_{pos:02d}_{variant}.png"), tile_gray)
        variant_summary = ", ".join(
            f"{item['variant']}={item['score']:.6f}@{item['loc']}" for item in match_info["variants"]
        )
        print(
            f"  pos_{pos:02d}: best={match_info['best_score']:.6f} ({match_info['best_variant']}) "
            f"loc={match_info['best_loc']} | {variant_summary}"
        )
    gate_pass = all(float(whole_scores.get(int(pos), 0.0)) >= float(args.threshold) for pos in args.positions)
    print(f"combined whole-ROI gate ({' AND '.join(f'best_{int(pos)}' for pos in args.positions)} >= threshold): {gate_pass}")

    print(f"fixed grid row ROI matches (x={TEMPLATE_GRID_X}, y={TEMPLATE_GRID_Y}, size={TEMPLATE_GRID_SIZE}) (best of black/white):")
    row_scores = {}
    for pos in args.positions:
        row_roi = row_search_roi_from_upscaled(upscaled, pos)
        row_roi_gray = cv2.cvtColor(row_roi, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(dump_dir / f"grid_row_{pos:02d}.png"), row_roi)
        match_info = _best_inverse_match(row_roi_gray, pos)
        row_scores[int(pos)] = float(match_info["best_score"])
        variant_summary = ", ".join(
            f"{item['variant']}={item['score']:.6f}@{item['loc']}" for item in match_info["variants"]
        )
        print(
            f"  row_{pos:02d}: best={match_info['best_score']:.6f} ({match_info['best_variant']}) "
            f"loc={match_info['best_loc']} shape={row_roi_gray.shape} | {variant_summary}"
        )
    layout_gate_pass = all(float(row_scores.get(int(pos), 0.0)) >= float(args.threshold) for pos in args.positions)
    print(f"  fixed-grid gate ({' AND '.join(f'best_{int(pos)}' for pos in args.positions)} >= threshold): {layout_gate_pass}")


if __name__ == "__main__":
    main()
