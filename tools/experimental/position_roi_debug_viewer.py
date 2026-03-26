"""Standalone experimental viewer for position-strip scan tuning.

This tool is intentionally isolated from the main GUI and runtime flow.
It lets you inspect a single frame, draw the 12 position ROIs, and review
the row-template scores plus the current pass/fail decision.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk

from mk8_local_play.extract_common import TARGET_HEIGHT, TARGET_WIDTH, crop_and_upscale_image, frame_to_timecode
from mk8_local_play.extract_initial_scan import (
    INITIAL_SCAN_TARGETS,
    _match_ignore_frame_target,
    _match_initial_scan_target,
    _expanded_roi,
    _bounded_roi,
    IGNORE_FRAME_TARGETS,
)
from mk8_local_play.data_paths import resolve_asset_file
from mk8_local_play.ocr_scoreboard_consensus import (
    POSITION_ROW_PADDING_BOTTOM,
    POSITION_ROW_PADDING_TOP,
    POSITION_ROW_PADDING_X,
    build_position_signal_metrics,
    position_strip_roi,
    position_template_row_windows,
    process_image,
)
from mk8_local_play.score_layouts import DEFAULT_SCORE_LAYOUT_ID, all_score_layouts


@dataclass
class RowDecision:
    row_number: int
    best_template: int
    score: float
    passes: bool


@dataclass
class ScoreLayoutEvaluation:
    layout_id: str
    decisions: list[RowDecision]
    average_score: float
    passed: bool


class PositionRoiDebugViewer:
    def __init__(self, root: tk.Tk, initial_video: str | None = None) -> None:
        self.root = root
        self.root.title("Experimental Position ROI Debug Viewer")
        self.root.geometry("1640x980")

        self.video_path: Path | None = Path(initial_video) if initial_video else None
        self.capture: cv2.VideoCapture | None = None
        self.templates = self._load_templates()
        self.frame_count = 0
        self.fps = 30.0
        self.current_frame = 0
        self.current_photo = None
        self.scan_autoplay = False
        self.last_scan_step = ""
        self.base_scan_skip_frames = 0
        self.last_scan_found_pass = False
        self.pending_resume_frame: int | None = None

        self.min_players_var = tk.IntVar(value=6)
        self.row_floor_var = tk.DoubleVar(value=0.40)
        self.avg_floor_var = tk.DoubleVar(value=0.60)
        self.frame_var = tk.StringVar(value="0")
        self.video_var = tk.StringVar(value=str(self.video_path) if self.video_path else "")
        self.status_var = tk.StringVar(value="Load a video to begin.")
        self.decision_var = tk.StringVar(value="")

        self._build_ui()
        if self.video_path:
            self._open_video(self.video_path)

    def _load_templates(self) -> list[tuple]:
        template_names = [
            "Score_template.png",
            "Trackname_template.png",
            "Race_template.png",
            "12th_pos_template.png",
            "ignore.png",
            "albumgallery_ignore.png",
            "ignore_2.png",
            "Race_template_NL_final.png",
        ]
        templates = []
        for template_name in template_names:
            template_path = resolve_asset_file("templates", template_name)
            template = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
            if template is None:
                raise FileNotFoundError(f"Template image could not be loaded: {template_path}")
            if len(template.shape) == 3 and template.shape[2] == 4:
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGRA2GRAY)
                _, alpha_mask = cv2.threshold(template[:, :, 3], 0, 255, cv2.THRESH_BINARY)
                _, template_binary = cv2.threshold(template_gray, 180, 255, cv2.THRESH_BINARY)
            elif len(template.shape) == 3 and template.shape[2] == 3:
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                _, template_binary = cv2.threshold(template_gray, 180, 255, cv2.THRESH_BINARY)
                alpha_mask = None
            elif len(template.shape) == 2:
                template_binary = template
                alpha_mask = None
            else:
                raise ValueError(f"Unexpected template channels for {template_path}")
            templates.append((template_binary, alpha_mask))
        return templates

    def _build_ui(self) -> None:
        root = self.root
        root.configure(bg="#151515")

        controls = tk.Frame(root, bg="#1f1f1f", padx=10, pady=10)
        controls.pack(side=tk.TOP, fill=tk.X)

        tk.Button(controls, text="Open Video", command=self._pick_video).grid(row=0, column=0, padx=4, pady=4, sticky="w")
        tk.Entry(controls, textvariable=self.video_var, width=90).grid(row=0, column=1, columnspan=8, padx=4, pady=4, sticky="ew")

        tk.Label(controls, text="Frame", bg="#1f1f1f", fg="white").grid(row=1, column=0, padx=4, pady=4, sticky="w")
        frame_entry = tk.Entry(controls, textvariable=self.frame_var, width=12)
        frame_entry.grid(row=1, column=1, padx=4, pady=4, sticky="w")
        frame_entry.bind("<Return>", lambda _event: self._jump_to_frame())
        tk.Button(controls, text="Go", command=self._jump_to_frame).grid(row=1, column=2, padx=4, pady=4, sticky="w")
        tk.Button(controls, text="-1", command=lambda: self._step_frame(-1)).grid(row=1, column=3, padx=2, pady=4)
        tk.Button(controls, text="+1", command=lambda: self._step_frame(1)).grid(row=1, column=4, padx=2, pady=4)
        tk.Button(controls, text="-10", command=lambda: self._step_frame(-10)).grid(row=1, column=5, padx=2, pady=4)
        tk.Button(controls, text="+10", command=lambda: self._step_frame(10)).grid(row=1, column=6, padx=2, pady=4)
        tk.Button(controls, text="-30", command=lambda: self._step_frame(-30)).grid(row=1, column=7, padx=2, pady=4)
        tk.Button(controls, text="+30", command=lambda: self._step_frame(30)).grid(row=1, column=8, padx=2, pady=4)

        tk.Label(controls, text="Min Players", bg="#1f1f1f", fg="white").grid(row=2, column=0, padx=4, pady=4, sticky="w")
        tk.Entry(controls, textvariable=self.min_players_var, width=8).grid(row=2, column=1, padx=4, pady=4, sticky="w")
        tk.Label(controls, text="Row Floor", bg="#1f1f1f", fg="white").grid(row=2, column=2, padx=4, pady=4, sticky="w")
        tk.Entry(controls, textvariable=self.row_floor_var, width=8).grid(row=2, column=3, padx=4, pady=4, sticky="w")
        tk.Label(controls, text="Average Floor", bg="#1f1f1f", fg="white").grid(row=2, column=4, padx=4, pady=4, sticky="w")
        tk.Entry(controls, textvariable=self.avg_floor_var, width=8).grid(row=2, column=5, padx=4, pady=4, sticky="w")
        tk.Button(controls, text="Recalculate", command=self._refresh_frame).grid(row=2, column=6, padx=4, pady=4, sticky="w")
        tk.Button(controls, text="Next Scan Frame", command=self._next_scan_frame).grid(row=2, column=7, padx=4, pady=4, sticky="w")
        self.autoplay_button = tk.Button(controls, text="Start Scan Play", command=self._toggle_scan_autoplay)
        self.autoplay_button.grid(row=2, column=8, padx=4, pady=4, sticky="w")

        self.slider = tk.Scale(
            controls,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            showvalue=False,
            length=900,
            command=self._on_slider,
            bg="#1f1f1f",
            fg="white",
            highlightthickness=0,
        )
        self.slider.grid(row=3, column=0, columnspan=9, padx=4, pady=8, sticky="ew")
        controls.grid_columnconfigure(1, weight=1)

        content = tk.Frame(root, bg="#151515")
        content.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        image_panel = tk.Frame(content, bg="#151515", padx=10, pady=10)
        image_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_label = tk.Label(image_panel, bg="#101010")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        side = tk.Frame(content, bg="#202020", width=480, padx=12, pady=12)
        side.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(side, text="Decision", bg="#202020", fg="white", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        tk.Label(side, textvariable=self.decision_var, justify=tk.LEFT, bg="#202020", fg="#dcdcdc", wraplength=430).pack(anchor="w", pady=(4, 12))

        tk.Label(side, text="Status", bg="#202020", fg="white", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        tk.Label(side, textvariable=self.status_var, justify=tk.LEFT, bg="#202020", fg="#dcdcdc", wraplength=430).pack(anchor="w", pady=(4, 12))

        tk.Label(side, text="Rows", bg="#202020", fg="white", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.rows_text = tk.Text(side, width=56, height=40, bg="#111111", fg="#e8e8e8", insertbackground="white")
        self.rows_text.pack(fill=tk.BOTH, expand=True)

    def _pick_video(self) -> None:
        selected = filedialog.askopenfilename(
            title="Open Video",
            filetypes=[("Video files", "*.mkv *.mp4 *.mov *.avi"), ("All files", "*.*")],
        )
        if selected:
            self._open_video(Path(selected))

    def _open_video(self, video_path: Path) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.video_path = video_path
        self.video_var.set(str(video_path))
        self.capture = cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            messagebox.showerror("Open failed", f"Could not open video:\n{video_path}")
            self.capture = None
            return
        self.frame_count = max(1, int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 30.0)
        self.base_scan_skip_frames = int(3 * max(1, int(self.fps)))
        self.slider.configure(to=max(0, self.frame_count - 1))
        self.current_frame = 0
        self.frame_var.set("0")
        self._refresh_frame()

    def _jump_to_frame(self) -> None:
        try:
            target = int(self.frame_var.get())
        except ValueError:
            return
        self.current_frame = max(0, min(target, max(0, self.frame_count - 1)))
        self._refresh_frame()

    def _step_frame(self, delta: int) -> None:
        self.current_frame = max(0, min(self.current_frame + int(delta), max(0, self.frame_count - 1)))
        self._refresh_frame()

    def _on_slider(self, value: str) -> None:
        try:
            self.current_frame = int(float(value))
        except ValueError:
            return
        self.frame_var.set(str(self.current_frame))
        self._refresh_frame(read_slider=False)

    def _refresh_frame(self, read_slider: bool = True) -> None:
        if self.capture is None or self.video_path is None:
            return
        if read_slider:
            self.slider.set(self.current_frame)
        self.frame_var.set(str(self.current_frame))

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ok, frame = self.capture.read()
        if not ok:
            self.status_var.set(f"Failed to read frame {self.current_frame}.")
            return

        upscaled = crop_and_upscale_image(frame, 0, 0, frame.shape[1], frame.shape[0], TARGET_WIDTH, TARGET_HEIGHT)
        decisions, scan_summary = self._evaluate_scan_frame(upscaled)
        annotated = self._draw_overlay(upscaled.copy(), decisions, scan_summary)
        self._show_image(annotated)

        timecode = frame_to_timecode(self.current_frame, self.fps)
        avg = sum(item.score for item in decisions[: max(0, self.min_players_var.get())]) / max(1, self.min_players_var.get())
        self.status_var.set(
            f"Video: {self.video_path.name}\n"
            f"Frame: {self.current_frame:,} / {max(0, self.frame_count - 1):,}\n"
            f"Timecode: {timecode}\n"
            f"Rows used: 1..{self.min_players_var.get()}"
        )
        self.decision_var.set(
            f"Score Pass: {'YES' if scan_summary['score_pass'] else 'NO'}\n"
            f"Score Layout: {scan_summary['score_layout_id']}\n"
            f"Per-row floor: {self.row_floor_var.get():.2f}\n"
            f"Average floor: {self.avg_floor_var.get():.2f}\n"
            f"Average rows 1..{self.min_players_var.get()}: {avg:.3f}\n"
            f"Ignore Pass: {'YES' if scan_summary['ignore_pass'] else 'NO'} | {scan_summary['ignore_label']} {scan_summary['ignore_score']:.3f} / {scan_summary['ignore_threshold']:.2f}\n"
            f"Track Pass: {'YES' if scan_summary['track_pass'] else 'NO'} | {scan_summary['track_score']:.3f} / {scan_summary['track_threshold']:.2f}\n"
            f"Race Pass: {'YES' if scan_summary['race_pass'] else 'NO'} | {scan_summary['race_score']:.3f} / {scan_summary['race_threshold']:.2f}\n"
            f"Simulated branch: {scan_summary['branch']}\n"
            f"{self.last_scan_step}"
        )
        self.rows_text.delete("1.0", tk.END)
        for item in decisions:
            self.rows_text.insert(
                tk.END,
                f"Row {item.row_number:02d} | best {item.best_template:02d} | score {item.score:.3f} | {'PASS' if item.passes else 'FAIL'}\n",
            )

    def _evaluate_rows(self, metrics: list[dict]) -> list[RowDecision]:
        decisions: list[RowDecision] = []
        min_players = max(2, int(self.min_players_var.get()))
        row_floor = float(self.row_floor_var.get())
        for row_number in range(1, 13):
            metric = metrics[row_number - 1] if row_number - 1 < len(metrics) else {}
            best_template = int(metric.get("best_position_template", 0))
            score = float(metric.get("best_position_score", 0.0))
            passes = True
            if row_number <= min_players:
                passes = best_template == row_number and score >= row_floor
            decisions.append(RowDecision(row_number, best_template, score, passes))
        return decisions

    def _passes(self, decisions: list[RowDecision]) -> bool:
        min_players = max(2, int(self.min_players_var.get()))
        avg_floor = float(self.avg_floor_var.get())
        required = decisions[:min_players]
        if not required or not all(item.passes for item in required):
            return False
        avg = sum(item.score for item in required) / len(required)
        return avg >= avg_floor

    def _evaluate_score_layouts(self, upscaled) -> ScoreLayoutEvaluation:
        evaluations: list[ScoreLayoutEvaluation] = []
        min_players = max(2, int(self.min_players_var.get()))
        for layout in all_score_layouts():
            processed = process_image(upscaled, score_layout_id=layout.layout_id)
            metrics = build_position_signal_metrics(processed, score_layout_id=layout.layout_id)
            decisions = self._evaluate_rows(metrics)
            required = decisions[:min_players]
            average_score = sum(item.score for item in required) / max(1, len(required))
            evaluations.append(
                ScoreLayoutEvaluation(
                    layout_id=layout.layout_id,
                    decisions=decisions,
                    average_score=average_score,
                    passed=self._passes(decisions),
                )
            )

        passing = [item for item in evaluations if item.passed]
        if passing:
            return max(passing, key=lambda item: item.average_score)
        return max(evaluations, key=lambda item: item.average_score, default=ScoreLayoutEvaluation(DEFAULT_SCORE_LAYOUT_ID, [], 0.0, False))

    def _evaluate_scan_frame(self, upscaled) -> tuple[list[RowDecision], dict]:
        gray_image = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        score_eval = self._evaluate_score_layouts(upscaled)
        decisions = score_eval.decisions
        score_pass = score_eval.passed
        score_value = score_eval.average_score if score_eval.decisions else 0.0

        stats = defaultdict(float)
        ignore_match = _match_ignore_frame_target(gray_image, self.templates, stats)
        track_target = next(target for target in INITIAL_SCAN_TARGETS if target["kind"] == "track")
        race_target = next(target for target in INITIAL_SCAN_TARGETS if target["kind"] == "race")
        track_score, track_blank, _track_roi = _match_initial_scan_target(gray_image, track_target, self.templates, stats)
        race_score, race_blank, _race_roi = _match_initial_scan_target(gray_image, race_target, self.templates, stats)
        race_candidate_matches = [
            {"roi": race_target["roi"], "template_index": int(race_target["template_index"])},
            *list(race_target.get("alternate_matches", ())),
        ]
        expanded_race_rois = [
            {
                "roi": _expanded_roi(gray_image, candidate["roi"]),
                "template_index": int(candidate["template_index"]),
            }
            for candidate in race_candidate_matches
        ]

        branch = "None"
        branch_skip_frames = 0
        uses_base_step = True
        if (
            not ignore_match["rejected_as_blank"]
            and ignore_match["max_val"] > ignore_match["match_threshold"]
        ):
            branch = ignore_match["label"]
            branch_skip_frames = int(round(self.fps * ignore_match["skip_seconds"]))
        else:
            for target in INITIAL_SCAN_TARGETS:
                if target["kind"] == "score":
                    if score_pass:
                        branch = target["label"]
                        branch_skip_frames = int(round(self.fps * target["skip_seconds"]))
                        break
                    continue
                target_score = track_score if target["kind"] == "track" else race_score
                target_blank = track_blank if target["kind"] == "track" else race_blank
                if target_blank:
                    if target["kind"] == "race":
                        branch = f"{target['label']} blank"
                        branch_skip_frames = 0
                        uses_base_step = False
                        break
                    continue
                if target_score > float(target["match_threshold"]):
                    branch = target["label"]
                    branch_skip_frames = int(round(self.fps * target["skip_seconds"]))
                    break

        total_advance_frames = branch_skip_frames + (self.base_scan_skip_frames if uses_base_step else 0)
        next_frame = self.current_frame if total_advance_frames == 0 else min(max(0, self.frame_count - 1), self.current_frame + max(1, total_advance_frames))
        next_timecode = frame_to_timecode(next_frame, self.fps)
        scan_summary = {
            "score_pass": score_pass,
            "score_value": score_value,
            "score_layout_id": score_eval.layout_id,
            "ignore_label": ignore_match["label"] or "Ignore",
            "ignore_score": float(ignore_match["max_val"]),
            "ignore_threshold": float(ignore_match["match_threshold"]),
            "ignore_pass": (
                not ignore_match["rejected_as_blank"]
                and float(ignore_match["max_val"]) > float(ignore_match["match_threshold"])
            ),
            "ignore_targets": [
                {
                    "label": str(target["label"]),
                    "roi": _bounded_roi(gray_image, target["roi"]),
                    "threshold": float(target["match_threshold"]),
                    "matched": str(target["label"]) == str(ignore_match["label"])
                    and not ignore_match["rejected_as_blank"]
                    and float(ignore_match["max_val"]) > float(target["match_threshold"]),
                }
                for target in IGNORE_FRAME_TARGETS
            ],
            "track_score": 0.0 if track_blank else float(track_score),
            "track_threshold": float(track_target["match_threshold"]),
            "track_pass": (not track_blank) and float(track_score) > float(track_target["match_threshold"]),
            "track_roi": _expanded_roi(gray_image, track_target["roi"]),
            "track_blank": bool(track_blank),
            "race_score": 0.0 if race_blank else float(race_score),
            "race_threshold": float(race_target["match_threshold"]),
            "race_pass": (not race_blank) and float(race_score) > float(race_target["match_threshold"]),
            "race_roi": expanded_race_rois[0]["roi"],
            "race_rois": expanded_race_rois,
            "race_blank": bool(race_blank),
            "branch": branch,
            "is_pause_branch": branch in {"Score", "TrackName", "RaceNumber"},
            "next_frame": next_frame,
            "next_timecode": next_timecode,
            "branch_skip_frames": branch_skip_frames,
            "base_scan_skip_frames": self.base_scan_skip_frames if uses_base_step else 0,
            "total_advance_frames": total_advance_frames,
        }
        return decisions, scan_summary

    def _next_scan_frame(self) -> None:
        if self.capture is None:
            return
        if self.pending_resume_frame is not None and self.pending_resume_frame != self.current_frame:
            resume_frame = max(0, min(self.pending_resume_frame, max(0, self.frame_count - 1)))
            self.pending_resume_frame = None
            self.last_scan_found_pass = False
            self.last_scan_step = f"Scan step: resumed from paused pass to frame {resume_frame} ({frame_to_timecode(resume_frame, self.fps)})"
            self.current_frame = resume_frame
            self._refresh_frame()
            return
        current_frame_before_step = self.current_frame
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ok, frame = self.capture.read()
        if not ok:
            return
        upscaled = crop_and_upscale_image(frame, 0, 0, frame.shape[1], frame.shape[0], TARGET_WIDTH, TARGET_HEIGHT)
        decisions, scan_summary = self._evaluate_scan_frame(upscaled)
        next_frame = int(scan_summary["next_frame"])
        self.last_scan_found_pass = bool(scan_summary["is_pause_branch"])
        self.last_scan_step = (
            f"Scan step: {scan_summary['branch']} -> branch skip {scan_summary['branch_skip_frames']} + "
            f"base {scan_summary['base_scan_skip_frames']} = {scan_summary['total_advance_frames']} frame(s) "
            f"to {next_frame} ({scan_summary['next_timecode']})"
        )
        if next_frame == self.current_frame:
            self.last_scan_step += " | no frame advance"
        if self.last_scan_found_pass:
            self.last_scan_step += f" | paused on detected {scan_summary['branch']} at frame {current_frame_before_step}"
            self.pending_resume_frame = next_frame
            self.current_frame = current_frame_before_step
        else:
            self.pending_resume_frame = None
            self.current_frame = next_frame
        self._refresh_frame()

    def _toggle_scan_autoplay(self) -> None:
        self.scan_autoplay = not self.scan_autoplay
        self.autoplay_button.configure(text="Stop Scan Play" if self.scan_autoplay else "Start Scan Play")
        if self.scan_autoplay:
            self._run_scan_autoplay()

    def _run_scan_autoplay(self) -> None:
        if not self.scan_autoplay:
            return
        previous_frame = self.current_frame
        self._next_scan_frame()
        if self.last_scan_found_pass or self.current_frame == previous_frame or self.current_frame >= max(0, self.frame_count - 1):
            self.scan_autoplay = False
            self.autoplay_button.configure(text="Start Scan Play")
            return
        self.root.after(120, self._run_scan_autoplay)

    def _draw_overlay(self, image, decisions: list[RowDecision], scan_summary: dict):
        score_layout_id = str(scan_summary.get("score_layout_id") or DEFAULT_SCORE_LAYOUT_ID)
        (x1, y1), (x2, y2) = position_strip_roi(score_layout_id=score_layout_id)
        for item, (start_y, end_y) in zip(decisions, position_template_row_windows()):
            crop_x1 = max(0, x1 - POSITION_ROW_PADDING_X)
            crop_y1 = max(0, y1 + start_y - POSITION_ROW_PADDING_TOP)
            crop_x2 = min(image.shape[1], x2 + POSITION_ROW_PADDING_X)
            crop_y2 = min(image.shape[0], y1 + end_y + POSITION_ROW_PADDING_BOTTOM)
            if item.row_number <= max(2, int(self.min_players_var.get())):
                color = (0, 255, 0) if item.passes else (0, 165, 255)
            else:
                color = (0, 0, 255)
            cv2.rectangle(image, (crop_x1, crop_y1), (crop_x2, crop_y2), color, 1)
            label = f"{item.row_number}:{item.score:.2f}/{item.best_template}"
            cv2.putText(
                image,
                label,
                (crop_x2 + 4, min(crop_y2 - 2, image.shape[0] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            image,
            f"Score Layout: {score_layout_id}",
            (12, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        for ignore_target in scan_summary.get("ignore_targets", []):
            roi_x, roi_y, roi_width, roi_height = ignore_target["roi"]
            color = (80, 80, 80)
            if ignore_target["matched"]:
                color = (255, 255, 0)
            cv2.rectangle(image, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), color, 1)
            cv2.putText(
                image,
                f"{ignore_target['label']}",
                (roi_x, max(12, roi_y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )

        track_color = (0, 255, 0) if scan_summary.get("track_pass") else ((255, 0, 255) if not scan_summary.get("track_blank") else (120, 120, 120))
        track_x, track_y, track_w, track_h = scan_summary["track_roi"]
        cv2.rectangle(image, (track_x, track_y), (track_x + track_w, track_y + track_h), track_color, 1)
        cv2.putText(
            image,
            f"Track {'PASS' if scan_summary['track_pass'] else 'FAIL'} {scan_summary['track_score']:.2f}/{scan_summary['track_threshold']:.2f}",
            (track_x, max(12, track_y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            track_color,
            1,
            cv2.LINE_AA,
        )

        race_color = (0, 255, 0) if scan_summary.get("race_pass") else ((0, 255, 255) if not scan_summary.get("race_blank") else (120, 120, 120))
        for index, race_info in enumerate(scan_summary.get("race_rois", [{"roi": scan_summary["race_roi"], "template_index": 2}]), start=1):
            race_x, race_y, race_w, race_h = race_info["roi"]
            outline_color = race_color if index == 1 else (180, 180, 0)
            cv2.rectangle(image, (race_x, race_y), (race_x + race_w, race_y + race_h), outline_color, 1)
            cv2.putText(
                image,
                f"Race ROI {index} T{race_info['template_index']}",
                (race_x, max(12, race_y - 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                outline_color,
                1,
                cv2.LINE_AA,
            )
        race_x, race_y, race_w, race_h = scan_summary["race_roi"]
        cv2.putText(
            image,
            f"Race {'PASS' if scan_summary['race_pass'] else 'FAIL'} {scan_summary['race_score']:.2f}/{scan_summary['race_threshold']:.2f}",
            (race_x, max(12, race_y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            race_color,
            1,
            cv2.LINE_AA,
        )
        return image

    def _show_image(self, image) -> None:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        self.current_photo = ImageTk.PhotoImage(pil_image)
        self.image_label.configure(image=self.current_photo)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental position ROI debug viewer")
    parser.add_argument("--video", help="Optional path to open immediately")
    args = parser.parse_args()

    root = tk.Tk()
    viewer = PositionRoiDebugViewer(root, initial_video=args.video)
    root.mainloop()


if __name__ == "__main__":
    main()
