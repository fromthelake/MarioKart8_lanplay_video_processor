"""Candidate-phase scan debugger for MK8 initial score candidate detection.

This tool is intentionally isolated from production flow.
It mirrors production initial-scan checks on production-checked frames and
explains why each candidate was (or was not) selected.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from mk8_local_play.data_paths import resolve_asset_file
from mk8_local_play.extract_common import TARGET_HEIGHT, TARGET_WIDTH, crop_and_upscale_image, frame_to_timecode
from mk8_local_play.extract_frames import _prepare_video_context
import mk8_local_play.extract_initial_scan as initial_scan
from mk8_local_play.extract_video_io import position_capture_for_read, read_video_frame
from mk8_local_play.project_paths import PROJECT_ROOT


def _load_templates():
    template_files = [
        "Trackname_template.png",
        "Race_template.png",
        "12th_pos_template.png",
        "ignore.png",
        "albumgallery_ignore.png",
        "ignore_2.png",
        "Race_template_NL_final.png",
        "12th_pos_templateNL.png",
    ]
    templates = []
    for filename in template_files:
        path = str(resolve_asset_file("templates", filename))
        template = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if template is None:
            raise RuntimeError(f"Template not found: {path}")
        if len(template.shape) == 3 and template.shape[2] == 4:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGRA2GRAY)
            _, alpha_mask = cv2.threshold(template[:, :, 3], 0, 255, cv2.THRESH_BINARY)
            _, template_binary = cv2.threshold(template_gray, 180, 255, cv2.THRESH_BINARY)
        elif len(template.shape) == 3 and template.shape[2] == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            _, template_binary = cv2.threshold(template_gray, 180, 255, cv2.THRESH_BINARY)
            alpha_mask = None
        else:
            template_binary = template
            alpha_mask = None
        templates.append((template_binary, alpha_mask))
    return templates


def _parse_pause_seconds(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        return 0.2
    return max(0.0, min(5.0, value))


def _target_threshold(kind: str) -> float:
    for target in initial_scan.INITIAL_SCAN_TARGETS:
        if str(target.get("kind")) == str(kind):
            return float(target.get("match_threshold", 0.0))
    return 0.0


def _target_skip_seconds(kind: str) -> float:
    for target in initial_scan.INITIAL_SCAN_TARGETS:
        if str(target.get("kind")) == str(kind):
            return float(target.get("skip_seconds", 0.0))
    return 0.0


def _extract_score_confirm_rows(layout_metrics: dict, layout_id: str, *, is_low_res_source=False) -> tuple[list[dict], list[float], float, bool]:
    metric = dict(layout_metrics.get(str(layout_id)) or {})
    position_metrics = list(metric.get("position_metrics") or [])
    thresholds = initial_scan._scan_thresholds(is_low_res_source=bool(is_low_res_source))
    row_coeff_threshold = float(thresholds["row_coeff"])
    row_start = int(initial_scan.INITIAL_SCAN_SCORE_PREFIX_ROW_START)
    row_end = int(initial_scan.POSITION_SCAN_MIN_PLAYERS)
    rows = []
    for row_number in range(row_start, row_end + 1):
        if row_number > len(position_metrics):
            break
        row_metric = position_metrics[row_number - 1]
        best_template = int(row_metric.get("best_position_template", 0))
        best_score = float(row_metric.get("best_position_score", 0.0))
        passed = bool(best_template == row_number and best_score >= row_coeff_threshold)
        rows.append(
            {
                "row": int(row_number),
                "best_template": int(best_template),
                "best_score": float(best_score),
                "pass": bool(passed),
            }
        )
    prefix_scores = [float(v) for v in list(metric.get("exact_prefix_scores") or [])]
    prefix_avg = float(metric.get("exact_prefix_average", 0.0) or 0.0)
    prefix_pass = bool(metric.get("exact_prefix_pass", False))
    return rows, prefix_scores, prefix_avg, prefix_pass


def build_candidate_trace(context, templates, progress_cb=None):
    def report(phase: str, percent: float):
        if progress_cb is None:
            return
        try:
            progress_cb(str(phase), float(percent))
        except Exception:
            pass

    fps = float(context["fps"])
    frame_skip = int(3 * max(1.0, fps))
    start_frame = int(context.get("usable_start_frame", 0) or 0)
    total_frames = int(context["total_frames"])
    end_frame = int(context.get("readable_end_frame", start_frame + total_frames) or (start_frame + total_frames))
    processing_video_path = str(context["processing_video_path"])
    left = int(context["median_left"])
    top = int(context["median_top"])
    crop_width = int(context["median_crop_width"])
    crop_height = int(context["median_crop_height"])
    is_low_res_source = bool(context.get("is_low_res_source", False))

    cap = cv2.VideoCapture(processing_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {processing_video_path}")

    stats = defaultdict(float)
    rows = []
    candidates = []
    checked_frames = []
    next_race_number = 1
    last_track_frame = 0
    last_race_frame = 0
    frame_count = int(start_frame)
    last_percent = -1
    total_span = max(1, end_frame - start_frame)

    try:
        report("Preparing candidate scan trace", 0.0)
        while frame_count < end_frame:
            progress = ((frame_count - start_frame) / float(total_span)) * 100.0
            progress_int = int(progress)
            if progress_int != last_percent and progress_int % 2 == 0:
                last_percent = progress_int
                report("Scanning production-checked frames", progress)

            if not position_capture_for_read(cap, frame_count, stats, max_forward_grab_frames=2, label="candidate_trace"):
                break
            ret, frame = read_video_frame(cap, stats)
            if not ret:
                break

            upscaled = crop_and_upscale_image(frame, left, top, crop_width, crop_height, TARGET_WIDTH, TARGET_HEIGHT)
            gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
            checked_frames.append(int(frame_count))

            row = {
                "frame": int(frame_count),
                "timecode": frame_to_timecode(int(frame_count), fps),
                "checked_by_code": True,
                "race_number_before_frame": int(next_race_number),
                "decision_kind": "advance_only",
                "decision_reason": "No trigger passed thresholds; continue regular 3s stride.",
                "branch_skip_seconds": 0.0,
                "branch_skip_frames": 0,
                "next_frame": int(frame_count + frame_skip),
            }

            ignore_match = initial_scan._match_ignore_frame_target(gray, templates, stats)
            row["ignore"] = {
                "label": str(ignore_match.get("label", "")),
                "score": float(ignore_match.get("max_val", 0.0)),
                "threshold": float(ignore_match.get("match_threshold", 0.0)),
                "pass": bool(
                    (not bool(ignore_match.get("rejected_as_blank", True)))
                    and float(ignore_match.get("max_val", 0.0)) > float(ignore_match.get("match_threshold", 0.0))
                    and not np.isinf(float(ignore_match.get("max_val", 0.0)))
                ),
                "rejected_as_blank": bool(ignore_match.get("rejected_as_blank", True)),
                "skip_seconds": float(ignore_match.get("skip_seconds", 0.0)),
            }

            if row["ignore"]["pass"]:
                skip_seconds = float(row["ignore"]["skip_seconds"])
                skip_frames = int(fps * skip_seconds)
                row["decision_kind"] = "ignore_skip"
                row["decision_reason"] = (
                    f"Ignore target '{row['ignore']['label']}' passed threshold, so scan skips forward {skip_seconds:.1f}s."
                )
                row["branch_skip_seconds"] = skip_seconds
                row["branch_skip_frames"] = int(skip_frames)
                row["next_frame"] = int(frame_count + frame_skip + skip_frames)
                rows.append(row)
                frame_count = int(row["next_frame"])
                continue

            score_threshold = float(_target_threshold("score"))
            track_threshold = float(_target_threshold("track"))
            race_threshold = float(_target_threshold("race"))
            score_entry = {
                "threshold": score_threshold,
                "gate_pass": False,
                "gate_layout_id": "",
                "gate_max": 0.0,
                "gate_row_scores": {},
                "gate_row_variants": {},
                "score_max": 0.0,
                "score_layout_id": "",
                "score_pass": False,
                "rejected_as_blank": False,
                "confirm_rows": [],
                "confirm_prefix_scores": [],
                "confirm_prefix_avg": 0.0,
                "confirm_prefix_pass": False,
            }
            track_entry = {
                "threshold": track_threshold,
                "max_val": 0.0,
                "pass": False,
                "rejected_as_blank": False,
                "saved_due_to_cooldown": False,
            }
            race_entry = {
                "threshold": race_threshold,
                "max_val": 0.0,
                "pass": False,
                "rejected_as_blank": False,
                "saved_due_to_cooldown": False,
            }

            decision_made = False
            for target in initial_scan.INITIAL_SCAN_TARGETS:
                kind = str(target.get("kind"))
                if kind == "score":
                    gate = initial_scan._initial_scan_score_gate(upscaled, stats)
                    gate_layout = str(gate.get("layout_id") or initial_scan.DEFAULT_SCORE_LAYOUT_ID)
                    score_entry["gate_pass"] = bool(gate.get("passed", False))
                    score_entry["gate_layout_id"] = gate_layout
                    score_entry["gate_max"] = float(gate.get("max_val", 0.0))
                    score_entry["gate_row_scores"] = {
                        int(k): float(v) for k, v in dict(gate.get("row_scores") or {}).items()
                    }
                    score_entry["gate_row_variants"] = {
                        int(k): str(v) for k, v in dict(gate.get("row_variants") or {}).items()
                    }
                    if not score_entry["gate_pass"]:
                        score_entry["score_max"] = 0.0
                        score_entry["score_layout_id"] = gate_layout
                        score_entry["rejected_as_blank"] = False
                    else:
                        max_val, rejected, layout_id, _metrics = initial_scan._match_score_target_layouts(
                            upscaled,
                            templates,
                            stats,
                            is_low_res_source=is_low_res_source,
                            return_layout_metrics=True,
                            preferred_layout_ids=[gate_layout],
                            stats_scope="scan",
                        )
                        score_entry["score_max"] = float(max_val)
                        score_entry["score_layout_id"] = str(layout_id or gate_layout)
                        score_entry["rejected_as_blank"] = bool(rejected)
                        confirm_rows, prefix_scores, prefix_avg, prefix_pass = _extract_score_confirm_rows(
                            _metrics,
                            score_entry["score_layout_id"],
                            is_low_res_source=is_low_res_source,
                        )
                        score_entry["confirm_rows"] = confirm_rows
                        score_entry["confirm_prefix_scores"] = prefix_scores
                        score_entry["confirm_prefix_avg"] = float(prefix_avg)
                        score_entry["confirm_prefix_pass"] = bool(prefix_pass)
                    score_entry["score_pass"] = bool(
                        (not bool(score_entry["rejected_as_blank"]))
                        and float(score_entry["score_max"]) > float(score_entry["threshold"])
                        and not np.isinf(float(score_entry["score_max"]))
                    )
                    row["score"] = score_entry
                    if score_entry["score_pass"]:
                        skip_seconds = float(_target_skip_seconds("score"))
                        skip_frames = int(fps * skip_seconds)
                        candidate = {
                            "race_number": int(next_race_number),
                            "frame_number": int(frame_count),
                            "timecode": frame_to_timecode(int(frame_count), fps),
                            "score_layout_id": str(score_entry["score_layout_id"] or initial_scan.DEFAULT_SCORE_LAYOUT_ID),
                            "score_value": float(score_entry["score_max"]),
                        }
                        candidates.append(candidate)
                        row["decision_kind"] = "score_candidate"
                        row["decision_reason"] = (
                            f"Score gate + score confirm passed; selected as race {next_race_number:03d} candidate."
                        )
                        row["branch_skip_seconds"] = skip_seconds
                        row["branch_skip_frames"] = int(skip_frames)
                        row["selected_race_number"] = int(next_race_number)
                        next_race_number += 1
                        row["next_frame"] = int(frame_count + frame_skip + skip_frames)
                        decision_made = True
                        break
                elif kind == "track":
                    max_val, rejected, _roi = initial_scan._match_initial_scan_target(gray, target, templates, stats)
                    track_entry["max_val"] = float(max_val)
                    track_entry["rejected_as_blank"] = bool(rejected)
                    track_entry["pass"] = bool(
                        (not bool(rejected))
                        and float(max_val) > float(track_threshold)
                        and not np.isinf(float(max_val))
                    )
                    row["track"] = track_entry
                    if track_entry["pass"]:
                        can_save = last_track_frame < max(1, int(frame_count) - int(fps * 20))
                        track_entry["saved_due_to_cooldown"] = bool(can_save)
                        if can_save:
                            last_track_frame = int(frame_count)
                        row["decision_kind"] = "track_detect"
                        row["decision_reason"] = (
                            "Track target passed threshold; production keeps 3s stride (no extra skip)."
                        )
                        row["next_frame"] = int(frame_count + frame_skip)
                        decision_made = True
                        break
                elif kind == "race":
                    max_val, rejected, _roi = initial_scan._match_initial_scan_target(gray, target, templates, stats)
                    race_entry["max_val"] = float(max_val)
                    race_entry["rejected_as_blank"] = bool(rejected)
                    if rejected:
                        row["race"] = race_entry
                        continue
                    race_entry["pass"] = bool(
                        float(max_val) > float(race_threshold) and not np.isinf(float(max_val))
                    )
                    row["race"] = race_entry
                    if race_entry["pass"]:
                        can_save = last_race_frame < max(1, int(frame_count) - int(fps * 20))
                        race_entry["saved_due_to_cooldown"] = bool(can_save)
                        if can_save:
                            last_race_frame = int(frame_count)
                        skip_seconds = float(_target_skip_seconds("race"))
                        skip_frames = int(fps * skip_seconds)
                        row["decision_kind"] = "race_detect"
                        row["decision_reason"] = (
                            "Race-number target passed threshold; scan applies race skip window."
                        )
                        row["branch_skip_seconds"] = skip_seconds
                        row["branch_skip_frames"] = int(skip_frames)
                        row["next_frame"] = int(frame_count + frame_skip + skip_frames)
                        decision_made = True
                        break

            if "score" not in row:
                row["score"] = score_entry
            if "track" not in row:
                row["track"] = track_entry
            if "race" not in row:
                row["race"] = race_entry

            if not decision_made:
                row["next_frame"] = int(frame_count + frame_skip)

            rows.append(row)
            frame_count = int(row["next_frame"])

        report("Finalizing candidate trace", 99.0)
        return {
            "rows": rows,
            "candidates": candidates,
            "checked_frames": checked_frames,
            "fps": fps,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frame_skip": frame_skip,
        }
    finally:
        report("Candidate trace ready", 100.0)
        cap.release()


class CandidateScanDebugGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MK8 Candidate Scan Debug GUI")
        self.geometry("1880x1020")

        self.templates = _load_templates()
        self.video_path = None
        self.context = None
        self.trace_data = None
        self.trace_rows = []
        self.trace_row_map = {}
        self.score_candidates = []
        self.checked_frames = []
        self.current_frame = 0
        self.range_start_frame = 0
        self.range_end_frame = 0
        self.is_playing = False
        self.is_production_playing = False
        self.production_play_btn_text = tk.StringVar(value="Play Production Frames")
        self.pause_seconds_var = tk.StringVar(value="0.20")
        self.jump_frame_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Select a video to start candidate scan debugging.")
        self.trace_progress_var = tk.StringVar(value="")
        self.trace_progress_value = tk.DoubleVar(value=0.0)
        self._trace_load_token = 0
        self._candidate_summary_entries = []
        self._summary_frame_links = {}
        self._frame_cache = {}
        self._inspection_row_cache = {}
        self._capture = None
        self._capture_stats = defaultdict(float)
        self.photo = None

        self._build_ui()

    def _is_low_res_source(self):
        if self.context is None:
            return False
        return bool(self.context.get("is_low_res_source", False))

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=8)
        ttk.Button(top, text="Select Video", command=self._on_select_video).pack(side=tk.LEFT)
        ttk.Button(top, text="Run Candidate Trace", command=self._run_trace).pack(side=tk.LEFT, padx=6)
        self.video_label = ttk.Label(top, text="No video selected", width=120)
        self.video_label.pack(side=tk.LEFT, padx=8)

        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        left = ttk.Frame(main, width=330)
        center = ttk.Frame(main)
        right = ttk.Frame(main, width=540)
        main.add(left, weight=1)
        main.add(center, weight=6)
        main.add(right, weight=2)

        ttk.Label(left, text="Detected Score Candidates").pack(anchor=tk.W)
        self.candidate_list = tk.Listbox(left, height=28)
        self.candidate_list.pack(fill=tk.BOTH, expand=True, pady=4)
        self.candidate_list.bind("<<ListboxSelect>>", self._on_candidate_selected)

        self.video_canvas = tk.Label(center, bg="#111111")
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        self.video_canvas.bind("<Configure>", lambda _e: self._render_current_frame())

        controls = ttk.Frame(center)
        controls.pack(fill=tk.X, pady=6)
        row1 = ttk.Frame(controls)
        row1.pack(fill=tk.X, pady=2)
        ttk.Button(row1, text="Frame -1", command=lambda: self._step_frames(-1)).pack(side=tk.LEFT)
        ttk.Button(row1, text="Frame +1", command=lambda: self._step_frames(1)).pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="Prod -1", command=lambda: self._step_production(-1)).pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="Prod +1", command=lambda: self._step_production(1)).pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="Jump -1s", command=lambda: self._step_seconds(-1.0)).pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="Jump +1s", command=lambda: self._step_seconds(1.0)).pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="Play/Pause", command=self._toggle_play).pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, textvariable=self.production_play_btn_text, command=self._toggle_play_production).pack(side=tk.LEFT, padx=4)

        row2 = ttk.Frame(controls)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="Jump to frame:").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.jump_frame_var, width=10).pack(side=tk.LEFT, padx=4)
        ttk.Button(row2, text="Go", command=self._jump_to_frame_from_entry).pack(side=tk.LEFT)
        ttk.Label(row2, text="Pause/frame (s):").pack(side=tk.LEFT, padx=(16, 4))
        ttk.Entry(row2, textvariable=self.pause_seconds_var, width=7).pack(side=tk.LEFT)

        ttk.Label(right, text="Summary").pack(anchor=tk.W)
        self.summary_text = tk.Text(right, height=12, wrap=tk.WORD)
        self.summary_text.pack(fill=tk.X, pady=(0, 6))
        self.summary_text.configure(state=tk.DISABLED)

        ttk.Label(right, text="Frame Metrics").pack(anchor=tk.W)
        self.metrics_text = tk.Text(right, height=46, wrap=tk.WORD)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        self.metrics_text.tag_config("bool_true", foreground="green")
        self.metrics_text.tag_config("bool_false", foreground="red")
        self.metrics_text.tag_config("bool_false_positive", foreground="#7bd88f")

        status = ttk.Label(self, textvariable=self.status_var)
        status.pack(fill=tk.X, padx=8, pady=(0, 4))
        trace_progress = ttk.Label(self, textvariable=self.trace_progress_var)
        trace_progress.pack(fill=tk.X, padx=8, pady=(0, 4))
        self.progress = ttk.Progressbar(self, mode="determinate", maximum=100.0, variable=self.trace_progress_value)
        self.progress.pack(fill=tk.X, padx=8, pady=(0, 6))

    def _on_select_video(self):
        initial_dir = str(PROJECT_ROOT / "Input_Videos")
        path = filedialog.askopenfilename(
            title="Select MK8 Video",
            initialdir=initial_dir,
            filetypes=[("Video files", "*.mp4 *.mkv *.avi *.mov"), ("All files", "*.*")],
        )
        if not path:
            return
        self.video_path = Path(path)
        self.video_label.configure(text=str(self.video_path))
        self.status_var.set("Video selected. Building video context...")
        self._prepare_context()

    def _prepare_context(self):
        if self.video_path is None:
            return
        folder_path = str(PROJECT_ROOT / "Input_Videos")
        context = _prepare_video_context(
            str(self.video_path),
            folder_path,
            True,
            1,
            1,
            0.0,
            self.templates,
            video_label=self.video_path.stem,
            source_display_name=str(self.video_path),
        )
        if context is None:
            raise RuntimeError("Failed to prepare video context")
        self.context = context
        self._open_capture()
        start_frame = int(context.get("usable_start_frame", 0) or 0)
        end_frame = int(context.get("readable_end_frame", start_frame + int(context["total_frames"])) or (start_frame + int(context["total_frames"])))
        self.range_start_frame = start_frame
        self.range_end_frame = max(start_frame, end_frame - 1)
        self.current_frame = start_frame
        self.jump_frame_var.set(str(self.current_frame))
        self._set_summary_lines(
            [
                ("Video", str(self.video_path.name), None),
                ("Start Frame", str(start_frame), start_frame),
                ("End Frame", str(self.range_end_frame), self.range_end_frame),
                ("FPS", f"{float(context['fps']):.3f}", None),
                ("Status", "Context prepared. Run Candidate Trace.", None),
            ]
        )
        self._render_current_frame()

    def _run_trace(self):
        if self.video_path is None:
            messagebox.showwarning("No video", "Select a video first.")
            return
        if self.context is None:
            self._prepare_context()
        self._trace_load_token += 1
        token = int(self._trace_load_token)
        self._set_trace_progress("starting", 0.0)
        self.status_var.set("Running candidate trace...")
        self.trace_rows = []
        self.trace_row_map = {}
        self.score_candidates = []
        self.checked_frames = []
        self._inspection_row_cache = {}
        self.candidate_list.delete(0, tk.END)

        def worker():
            try:
                def progress_update(phase, percent):
                    self.after(0, lambda: self._set_trace_progress(phase, percent))

                trace = build_candidate_trace(self.context, self.templates, progress_cb=progress_update)
                self.after(0, lambda: self._on_trace_loaded(token, trace))
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("Trace failed", str(exc)))
                self.after(0, lambda: self.status_var.set("Candidate trace failed."))
                self.after(0, lambda: self._set_trace_progress("failed", 0.0))

        threading.Thread(target=worker, daemon=True).start()

    def _on_trace_loaded(self, token, trace):
        if int(token) != int(self._trace_load_token):
            return
        self.trace_data = trace
        self.trace_rows = list(trace.get("rows") or [])
        self.trace_row_map = {int(row["frame"]): row for row in self.trace_rows}
        self.score_candidates = list(trace.get("candidates") or [])
        self.checked_frames = sorted(int(v) for v in trace.get("checked_frames") or [])
        self.candidate_list.delete(0, tk.END)
        for candidate in self.score_candidates:
            race_num = int(candidate.get("race_number", 0))
            frame_num = int(candidate.get("frame_number", 0))
            value = float(candidate.get("score_value", 0.0))
            layout_id = str(candidate.get("score_layout_id", ""))
            label = f"Race {race_num:03d} | Frame {frame_num} | {frame_to_timecode(frame_num, float(trace.get('fps', 30.0)))} | score {value:.3f} | {layout_id}"
            self.candidate_list.insert(tk.END, label)
        if self.checked_frames:
            self.current_frame = int(self.checked_frames[0])
            self.range_start_frame = int(self.checked_frames[0])
            self.range_end_frame = int(self.checked_frames[-1])
            self.jump_frame_var.set(str(self.current_frame))
            self._render_current_frame()
        self._set_summary_lines(
            [
                ("Checked Frames", str(len(self.checked_frames)), None),
                ("Detected Candidates", str(len(self.score_candidates)), None),
                ("First Checked Frame", str(self.checked_frames[0] if self.checked_frames else "None"), self.checked_frames[0] if self.checked_frames else None),
                ("Last Checked Frame", str(self.checked_frames[-1] if self.checked_frames else "None"), self.checked_frames[-1] if self.checked_frames else None),
                ("Status", "Trace ready. Select a candidate for focused review.", None),
            ]
        )
        self.status_var.set("Candidate trace loaded.")
        self._set_trace_progress("ready", 100.0)

    def _on_candidate_selected(self, _event=None):
        index = self.candidate_list.curselection()
        if not index or not self.score_candidates:
            return
        candidate = self.score_candidates[int(index[0])]
        candidate_frame = int(candidate.get("frame_number", 0))
        fps = float(self.context["fps"]) if self.context else 30.0
        self.current_frame = candidate_frame
        self.jump_frame_var.set(str(self.current_frame))
        self._render_current_frame()

        checked_index = -1
        if candidate_frame in self.checked_frames:
            checked_index = self.checked_frames.index(candidate_frame)
        prev_checked = self.checked_frames[checked_index - 1] if checked_index > 0 else None
        next_checked = self.checked_frames[checked_index + 1] if checked_index >= 0 and checked_index + 1 < len(self.checked_frames) else None
        prev_candidate = None
        next_candidate = None
        for item in self.score_candidates:
            frame = int(item.get("frame_number", 0))
            if frame < candidate_frame:
                prev_candidate = frame
            elif frame > candidate_frame and next_candidate is None:
                next_candidate = frame
        window_start = max(self.range_start_frame, candidate_frame - int(6 * fps))
        window_end = min(self.range_end_frame, candidate_frame + int(6 * fps))
        self._set_summary_lines(
            [
                ("Race", f"{int(candidate.get('race_number', 0)):03d}", None),
                ("Candidate Frame", str(candidate_frame), candidate_frame),
                ("Candidate Time", frame_to_timecode(candidate_frame, fps), None),
                ("Score Value", f"{float(candidate.get('score_value', 0.0)):.4f}", None),
                ("Score Layout", str(candidate.get("score_layout_id", "")), None),
                ("Previous Checked Frame", str(prev_checked), prev_checked),
                ("Next Checked Frame", str(next_checked), next_checked),
                ("Previous Candidate", str(prev_candidate), prev_candidate),
                ("Next Candidate", str(next_candidate), next_candidate),
                ("Focus Range Start (-6s)", str(window_start), window_start),
                ("Focus Range End (+6s)", str(window_end), window_end),
            ]
        )
        self.status_var.set(
            f"Selected race {int(candidate.get('race_number', 0)):03d}. Candidate frame {candidate_frame}."
        )

    def _set_trace_progress(self, phase, percent):
        clamped = max(0.0, min(100.0, float(percent)))
        self.trace_progress_var.set(f"Candidate trace: {phase} ({clamped:.0f}%)")
        self.trace_progress_value.set(clamped)

    def _set_summary_lines(self, entries):
        self._candidate_summary_entries = list(entries)
        self._summary_frame_links = {}
        self.summary_text.configure(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)
        for idx, (label, value, frame_ref) in enumerate(self._candidate_summary_entries):
            self.summary_text.insert(tk.END, f"{label}: ")
            value_text = str(value)
            if frame_ref is None:
                self.summary_text.insert(tk.END, f"{value_text}\n")
                continue
            tag = f"jump_{idx}"
            self.summary_text.insert(tk.END, f"{value_text}\n", (tag,))
            self._summary_frame_links[tag] = frame_ref
            self.summary_text.tag_config(tag, foreground="#2a6fdb", underline=1)
            self.summary_text.tag_bind(tag, "<Button-1>", lambda _e, t=tag: self._on_summary_click(t))
        self.summary_text.configure(state=tk.DISABLED)

    def _on_summary_click(self, tag):
        frame_ref = self._summary_frame_links.get(tag)
        if frame_ref is None:
            return
        if isinstance(frame_ref, list):
            if not frame_ref:
                return
            target = int(frame_ref[0])
        else:
            target = int(frame_ref)
        self.current_frame = max(0, int(target))
        self.jump_frame_var.set(str(self.current_frame))
        self._render_current_frame()

    def _open_capture(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._frame_cache = {}
        self._inspection_row_cache = {}
        if self.context is None:
            return
        self._capture = cv2.VideoCapture(str(self.context["processing_video_path"]))
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open video: {self.context['processing_video_path']}")

    def _get_upscaled_frame(self, frame_number):
        key = int(frame_number)
        if key in self._frame_cache:
            return self._frame_cache[key].copy()
        if self._capture is None:
            self._open_capture()
        if self._capture is None:
            return None
        if not position_capture_for_read(self._capture, int(frame_number), self._capture_stats, max_forward_grab_frames=0, label="gui_render"):
            return None
        ret, frame = read_video_frame(self._capture, self._capture_stats)
        if not ret:
            return None
        upscaled = crop_and_upscale_image(
            frame,
            int(self.context["median_left"]),
            int(self.context["median_top"]),
            int(self.context["median_crop_width"]),
            int(self.context["median_crop_height"]),
            TARGET_WIDTH,
            TARGET_HEIGHT,
        )
        if len(self._frame_cache) >= 24:
            self._frame_cache.pop(next(iter(self._frame_cache)))
        self._frame_cache[key] = upscaled.copy()
        return upscaled

    def _estimate_race_index_for_frame(self, frame_number):
        race_index = 1
        for item in self.score_candidates:
            if int(item.get("frame_number", -1)) <= int(frame_number):
                race_index += 1
            else:
                break
        return int(race_index)

    def _evaluate_frame_for_overlay(self, upscaled, frame_number):
        key = int(frame_number)
        if key in self._inspection_row_cache:
            return dict(self._inspection_row_cache[key])

        stats = defaultdict(float)
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

        ignore_match = initial_scan._match_ignore_frame_target(gray, self.templates, stats)
        ignore_pass = bool(
            (not bool(ignore_match.get("rejected_as_blank", True)))
            and float(ignore_match.get("max_val", 0.0)) > float(ignore_match.get("match_threshold", 0.0))
            and not np.isinf(float(ignore_match.get("max_val", 0.0)))
        )
        score_threshold = float(_target_threshold("score"))
        gate = initial_scan._initial_scan_score_gate(upscaled, stats)
        gate_layout = str(gate.get("layout_id") or initial_scan.DEFAULT_SCORE_LAYOUT_ID)
        if bool(gate.get("passed", False)):
            max_val, rejected, layout_id, _metrics = initial_scan._match_score_target_layouts(
                upscaled,
                self.templates,
                stats,
                is_low_res_source=bool(self._is_low_res_source()),
                return_layout_metrics=True,
                preferred_layout_ids=[gate_layout],
                stats_scope="scan",
            )
            score_layout_id = str(layout_id or gate_layout)
            score_max = float(max_val)
            score_rejected = bool(rejected)
            confirm_rows, prefix_scores, prefix_avg, prefix_pass = _extract_score_confirm_rows(
                _metrics,
                score_layout_id,
                is_low_res_source=bool(self._is_low_res_source()),
            )
        else:
            score_layout_id = gate_layout
            score_max = 0.0
            score_rejected = False
            confirm_rows, prefix_scores, prefix_avg, prefix_pass = ([], [], 0.0, False)
        score_pass = bool((not score_rejected) and score_max > score_threshold and not np.isinf(score_max))

        track_target = next((t for t in initial_scan.INITIAL_SCAN_TARGETS if str(t.get("kind")) == "track"), None)
        race_target = next((t for t in initial_scan.INITIAL_SCAN_TARGETS if str(t.get("kind")) == "race"), None)
        track_threshold = float(_target_threshold("track"))
        race_threshold = float(_target_threshold("race"))
        track_max = 0.0
        track_rejected = False
        race_max = 0.0
        race_rejected = False
        if track_target is not None:
            track_max, track_rejected, _roi = initial_scan._match_initial_scan_target(gray, track_target, self.templates, stats)
        if race_target is not None:
            race_max, race_rejected, _roi = initial_scan._match_initial_scan_target(gray, race_target, self.templates, stats)
        track_pass = bool((not track_rejected) and float(track_max) > track_threshold and not np.isinf(float(track_max)))
        race_pass = bool((not race_rejected) and float(race_max) > race_threshold and not np.isinf(float(race_max)))

        row = {
            "frame": int(frame_number),
            "timecode": frame_to_timecode(int(frame_number), float(self.context["fps"])),
            "checked_by_code": False,
            "race_number_before_frame": int(self._estimate_race_index_for_frame(frame_number)),
            "decision_kind": "inspection_only",
            "decision_reason": "This frame is not part of production scan stride; values shown for comparison only.",
            "branch_skip_seconds": 0.0,
            "branch_skip_frames": 0,
            "next_frame": int(frame_number + 1),
            "ignore": {
                "label": str(ignore_match.get("label", "")),
                "score": float(ignore_match.get("max_val", 0.0)),
                "threshold": float(ignore_match.get("match_threshold", 0.0)),
                "pass": bool(ignore_pass),
                "rejected_as_blank": bool(ignore_match.get("rejected_as_blank", True)),
                "skip_seconds": float(ignore_match.get("skip_seconds", 0.0)),
            },
            "score": {
                "threshold": float(score_threshold),
                "gate_pass": bool(gate.get("passed", False)),
                "gate_layout_id": gate_layout,
                "gate_max": float(gate.get("max_val", 0.0)),
                "gate_row_scores": {int(k): float(v) for k, v in dict(gate.get("row_scores") or {}).items()},
                "gate_row_variants": {int(k): str(v) for k, v in dict(gate.get("row_variants") or {}).items()},
                "score_max": float(score_max),
                "score_layout_id": score_layout_id,
                "score_pass": bool(score_pass),
                "rejected_as_blank": bool(score_rejected),
                "confirm_rows": list(confirm_rows),
                "confirm_prefix_scores": list(prefix_scores),
                "confirm_prefix_avg": float(prefix_avg),
                "confirm_prefix_pass": bool(prefix_pass),
            },
            "track": {
                "threshold": float(track_threshold),
                "max_val": float(track_max),
                "pass": bool(track_pass),
                "rejected_as_blank": bool(track_rejected),
                "saved_due_to_cooldown": False,
            },
            "race": {
                "threshold": float(race_threshold),
                "max_val": float(race_max),
                "pass": bool(race_pass),
                "rejected_as_blank": bool(race_rejected),
                "saved_due_to_cooldown": False,
            },
        }
        if len(self._inspection_row_cache) >= 64:
            self._inspection_row_cache.pop(next(iter(self._inspection_row_cache)))
        self._inspection_row_cache[key] = dict(row)
        return row

    def _render_current_frame(self):
        if self.context is None:
            return
        self.current_frame = max(0, int(self.current_frame))
        self.jump_frame_var.set(str(self.current_frame))
        frame = self._get_upscaled_frame(self.current_frame)
        if frame is None:
            self.status_var.set(f"Failed to read frame {self.current_frame}")
            return

        production_row = self.trace_row_map.get(int(self.current_frame))
        row = production_row
        if row is None:
            row = self._evaluate_frame_for_overlay(frame, int(self.current_frame))
        annotated = frame.copy()
        annotated = self._draw_overlay(annotated, row)
        self._render_metrics(row)
        if bool(production_row is not None):
            self.status_var.set(
                f"Frame {self.current_frame} ({frame_to_timecode(self.current_frame, float(self.context['fps']))}) | decision: {row.get('decision_kind', 'n/a')}"
            )
        else:
            self.status_var.set(
                f"Frame {self.current_frame} ({frame_to_timecode(self.current_frame, float(self.context['fps']))}) | inspection mode (not production-checked)."
            )

        self._show_image(annotated)

    def _draw_overlay(self, image, row):
        height, width = image.shape[:2]
        _ = (height, width)

        # Draw ignore ROIs and highlight the matched ignore label.
        ignore_info = row.get("ignore", {})
        matched_label = str(ignore_info.get("label", ""))
        for target in initial_scan.IGNORE_FRAME_TARGETS:
            rx, ry, rw, rh = initial_scan._bounded_roi(image, target["roi"])
            label = str(target.get("label", ""))
            is_match = label == matched_label
            color = (0, 255, 255) if is_match else (130, 130, 130)
            cv2.rectangle(image, (rx, ry), (rx + rw, ry + rh), color, 1)
            cv2.putText(
                image,
                f"{label}{' *' if is_match else ''}",
                (rx, max(14, ry - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )

        # Track and race ROIs.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        track_target = next((t for t in initial_scan.INITIAL_SCAN_TARGETS if str(t.get("kind")) == "track"), None)
        race_target = next((t for t in initial_scan.INITIAL_SCAN_TARGETS if str(t.get("kind")) == "race"), None)
        if track_target is not None:
            tx, ty, tw, th = initial_scan._expanded_roi(gray, track_target["roi"])
            track_pass = bool((row.get("track") or {}).get("pass", False))
            color = (0, 255, 0) if track_pass else (140, 140, 140)
            cv2.rectangle(image, (tx, ty), (tx + tw, ty + th), color, 1)
            cv2.putText(image, "Track ROI", (tx, max(14, ty - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        if race_target is not None:
            rx, ry, rw, rh = initial_scan._expanded_roi(gray, race_target["roi"])
            race_pass = bool((row.get("race") or {}).get("pass", False))
            color = (0, 255, 0) if race_pass else (140, 140, 140)
            cv2.rectangle(image, (rx, ry), (rx + rw, ry + rh), color, 1)
            cv2.putText(image, "Race ROI", (rx, max(14, ry - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        # Score gate row tiles.
        score = row.get("score") or {}
        gate_layout = str(score.get("gate_layout_id") or initial_scan.DEFAULT_SCORE_LAYOUT_ID)
        gate_scores = dict(score.get("gate_row_scores") or {})
        row_start = int(initial_scan.INITIAL_SCAN_GATE_ROW_START)
        row_end = int(initial_scan.INITIAL_SCAN_GATE_ROW_END)
        for row_number in range(row_start, row_end + 1):
            tile = initial_scan._initial_scan_gate_tile_roi(image, row_number, score_layout_id=gate_layout)
            tile_h, tile_w = tile.shape[:2]
            x1 = int(initial_scan.INITIAL_SCAN_GATE_TEMPLATE_GRID_X)
            if gate_layout == initial_scan.LAN1_SCORE_LAYOUT_ID:
                x1 += int(initial_scan.SCORE_LAYOUT_SHIFT_X)
            y1 = int(initial_scan.INITIAL_SCAN_GATE_TEMPLATE_GRID_Y) + ((int(row_number) - 1) * int(initial_scan.INITIAL_SCAN_GATE_TEMPLATE_GRID_SIZE))
            coeff = float(gate_scores.get(int(row_number), 0.0))
            passed = coeff >= float(initial_scan.INITIAL_SCAN_GATE_MIN_COEFF)
            color = (0, 255, 0) if passed else (0, 140, 255)
            cv2.rectangle(image, (x1, y1), (x1 + tile_w, y1 + tile_h), color, 1)
            cv2.putText(
                image,
                f"R{row_number}:{coeff:.3f}",
                (x1, max(14, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )

        # Score confirm rows (usually rows 2-6) so failed prefix rows are visible.
        confirm_rows = list(score.get("confirm_rows") or [])
        tile_size = int(initial_scan.INITIAL_SCAN_GATE_TEMPLATE_GRID_SIZE)
        base_x = int(initial_scan.INITIAL_SCAN_GATE_TEMPLATE_GRID_X)
        if gate_layout == initial_scan.LAN1_SCORE_LAYOUT_ID:
            base_x += int(initial_scan.SCORE_LAYOUT_SHIFT_X)
        base_y = int(initial_scan.INITIAL_SCAN_GATE_TEMPLATE_GRID_Y)
        for item in confirm_rows:
            row_number = int(item.get("row", 0))
            if row_number <= 0:
                continue
            x1 = int(base_x)
            y1 = int(base_y + ((row_number - 1) * tile_size))
            passed = bool(item.get("pass", False))
            color = (0, 200, 0) if passed else (0, 80, 255)
            cv2.rectangle(image, (x1 + 2, y1 + 2), (x1 + tile_size - 2, y1 + tile_size - 2), color, 1)
            cv2.putText(
                image,
                f"C{row_number}:{float(item.get('best_score', 0.0)):.3f}/T{int(item.get('best_template', 0))}",
                (x1 + tile_size + 4, y1 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )

        # Header.
        decision_kind = str(row.get("decision_kind", "advance_only"))
        scan_mode = "production" if bool(row.get("checked_by_code", False)) else "inspection"
        text = f"Frame {int(row.get('frame', 0))} | Mode: {scan_mode} | Decision: {decision_kind} | Next: {int(row.get('next_frame', 0))}"
        cv2.putText(image, text, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    def _render_metrics(self, row):
        self.metrics_text.configure(state=tk.NORMAL)
        self.metrics_text.delete("1.0", tk.END)

        def write_bool(label, value):
            self.metrics_text.insert(tk.END, f"{label}: ")
            text = "TRUE" if bool(value) else "FALSE"
            tag = "bool_true" if bool(value) else "bool_false"
            self.metrics_text.insert(tk.END, text + "\n", (tag,))

        def write_bool_with_intent(label, value, *, false_is_positive=False):
            self.metrics_text.insert(tk.END, f"{label}: ")
            text = "TRUE" if bool(value) else "FALSE"
            if bool(value):
                tag = "bool_true"
            else:
                tag = "bool_false_positive" if bool(false_is_positive) else "bool_false"
            self.metrics_text.insert(tk.END, text + "\n", (tag,))

        self.metrics_text.insert(tk.END, "Goal\n")
        self.metrics_text.insert(
            tk.END,
            "Find score screens reliably. If score passes, create a race candidate and skip ahead by production rules.\n\n",
        )
        if not bool(row.get("checked_by_code", False)):
            self.metrics_text.insert(
                tk.END,
                "Note: This is an inspection frame (not checked in production stride). Values are computed live for comparison.\n\n",
            )
        self.metrics_text.insert(tk.END, "Production Step Outcome\n")
        self.metrics_text.insert(tk.END, f"- Frame: {int(row.get('frame', 0))} ({row.get('timecode', '')})\n")
        self.metrics_text.insert(tk.END, f"- Race index before frame: {int(row.get('race_number_before_frame', 0)):03d}\n")
        self.metrics_text.insert(tk.END, f"- Decision: {row.get('decision_kind', '')}\n")
        self.metrics_text.insert(tk.END, f"- Why: {row.get('decision_reason', '')}\n")
        self.metrics_text.insert(tk.END, f"- Next checked frame: {int(row.get('next_frame', 0))}\n")
        self.metrics_text.insert(tk.END, f"- Branch skip: {float(row.get('branch_skip_seconds', 0.0)):.1f}s ({int(row.get('branch_skip_frames', 0))} frames)\n\n")

        ignore = row.get("ignore") or {}
        self.metrics_text.insert(tk.END, "Check 1: Ignore screens\n")
        write_bool_with_intent("- Passed ignore check", ignore.get("pass", False), false_is_positive=True)
        self.metrics_text.insert(tk.END, f"- Ignore label: {ignore.get('label', '')}\n")
        self.metrics_text.insert(tk.END, f"- Ignore score: {float(ignore.get('score', 0.0)):.4f}\n")
        self.metrics_text.insert(tk.END, f"- Ignore threshold: {float(ignore.get('threshold', 0.0)):.4f}\n\n")

        score = row.get("score") or {}
        self.metrics_text.insert(tk.END, "Check 2: Score gate and score confirm\n")
        write_bool("- Gate pass", score.get("gate_pass", False))
        self.metrics_text.insert(tk.END, f"- Gate layout: {score.get('gate_layout_id', '')}\n")
        self.metrics_text.insert(tk.END, f"- Gate max row score: {float(score.get('gate_max', 0.0)):.4f}\n")
        write_bool("- Score confirm pass", score.get("score_pass", False))
        self.metrics_text.insert(tk.END, f"- Score value: {float(score.get('score_max', 0.0)):.4f}\n")
        self.metrics_text.insert(tk.END, f"- Score threshold: {float(score.get('threshold', 0.0)):.4f}\n")
        self.metrics_text.insert(tk.END, f"- Score layout: {score.get('score_layout_id', '')}\n")
        self.metrics_text.insert(
            tk.END,
            f"- Confirm prefix avg (rows {int(initial_scan.INITIAL_SCAN_SCORE_PREFIX_ROW_START)}-{int(initial_scan.POSITION_SCAN_MIN_PLAYERS)}): "
            f"{float(score.get('confirm_prefix_avg', 0.0)):.4f}\n",
        )
        write_bool("- Confirm prefix pass", score.get("confirm_prefix_pass", False))
        confirm_rows = list(score.get("confirm_rows") or [])
        if confirm_rows:
            self.metrics_text.insert(tk.END, "  Score confirm rows:\n")
            for item in confirm_rows:
                self.metrics_text.insert(
                    tk.END,
                    f"  - Row {int(item.get('row', 0))}: "
                    f"best_template={int(item.get('best_template', 0))}, "
                    f"score={float(item.get('best_score', 0.0)):.4f}, pass=",
                )
                pass_tag = "bool_true" if bool(item.get("pass", False)) else "bool_false"
                self.metrics_text.insert(
                    tk.END,
                    "TRUE\n" if bool(item.get("pass", False)) else "FALSE\n",
                    (pass_tag,),
                )
        row_scores = score.get("gate_row_scores") or {}
        if row_scores:
            self.metrics_text.insert(tk.END, "  Gate row scores:\n")
            for key in sorted(row_scores.keys()):
                self.metrics_text.insert(tk.END, f"  - Row {int(key)}: {float(row_scores[key]):.4f}\n")
        self.metrics_text.insert(tk.END, "\n")

        track = row.get("track") or {}
        self.metrics_text.insert(tk.END, "Check 3: Track target\n")
        write_bool_with_intent("- Track pass", track.get("pass", False), false_is_positive=True)
        self.metrics_text.insert(tk.END, f"- Track score: {float(track.get('max_val', 0.0)):.4f}\n")
        self.metrics_text.insert(tk.END, f"- Track threshold: {float(track.get('threshold', 0.0)):.4f}\n")
        write_bool("- Track saved (20s cooldown)", track.get("saved_due_to_cooldown", False))
        self.metrics_text.insert(tk.END, "\n")

        race = row.get("race") or {}
        self.metrics_text.insert(tk.END, "Check 4: Race number target\n")
        write_bool_with_intent("- Race pass", race.get("pass", False), false_is_positive=True)
        self.metrics_text.insert(tk.END, f"- Race score: {float(race.get('max_val', 0.0)):.4f}\n")
        self.metrics_text.insert(tk.END, f"- Race threshold: {float(race.get('threshold', 0.0)):.4f}\n")
        write_bool("- Race saved (20s cooldown)", race.get("saved_due_to_cooldown", False))
        self.metrics_text.insert(tk.END, "\n")

        self.metrics_text.configure(state=tk.DISABLED)

    def _show_image(self, image):
        h, w = image.shape[:2]
        container_w = max(1, int(self.video_canvas.winfo_width()))
        container_h = max(1, int(self.video_canvas.winfo_height()))
        if container_w <= 1 or container_h <= 1:
            container_w, container_h = 1280, 720
        scale = min(container_w / max(1, w), container_h / max(1, h))
        resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        self.photo = ImageTk.PhotoImage(pil_image)
        self.video_canvas.configure(image=self.photo)

    def _step_frames(self, delta):
        self.current_frame = max(0, int(self.current_frame) + int(delta))
        self._render_current_frame()

    def _step_seconds(self, seconds):
        if self.context is None:
            return
        fps = float(self.context["fps"])
        delta = int(round(float(seconds) * fps))
        self._step_frames(delta)

    def _toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.is_production_playing = False
            self.production_play_btn_text.set("Play Production Frames")
            self._play_loop()

    def _play_loop(self):
        if not self.is_playing:
            return
        self._step_frames(1)
        delay_ms = max(1, int(_parse_pause_seconds(self.pause_seconds_var.get()) * 1000))
        self.after(delay_ms, self._play_loop)

    def _toggle_play_production(self):
        self.is_production_playing = not self.is_production_playing
        self.production_play_btn_text.set("Pause Production Frames" if self.is_production_playing else "Play Production Frames")
        if self.is_production_playing:
            self.is_playing = False
            self._play_production_loop()

    def _step_production(self, direction):
        if not self.checked_frames:
            self.status_var.set("No production-checked frames available. Run Candidate Trace first.")
            return
        current = int(self.current_frame)
        target = None
        if int(direction) >= 0:
            for frame in self.checked_frames:
                if int(frame) > current:
                    target = int(frame)
                    break
            if target is None:
                target = int(self.checked_frames[0])
        else:
            for frame in reversed(self.checked_frames):
                if int(frame) < current:
                    target = int(frame)
                    break
            if target is None:
                target = int(self.checked_frames[-1])
        self.current_frame = int(target)
        self._render_current_frame()

    def _play_production_loop(self):
        if not self.is_production_playing:
            return
        if not self.checked_frames:
            self.status_var.set("No production-checked frames available. Run Candidate Trace first.")
            self.is_production_playing = False
            self.production_play_btn_text.set("Play Production Frames")
            return
        self._step_production(1)
        delay_ms = max(1, int(_parse_pause_seconds(self.pause_seconds_var.get()) * 1000))
        self.after(delay_ms, self._play_production_loop)

    def _jump_to_frame_from_entry(self):
        value = str(self.jump_frame_var.get()).strip()
        if not value:
            return
        try:
            frame = int(value)
        except ValueError:
            messagebox.showwarning("Invalid frame", "Enter a valid integer frame number.")
            return
        self.current_frame = max(0, frame)
        self._render_current_frame()

    def destroy(self):
        try:
            if self._capture is not None:
                self._capture.release()
        except Exception:
            pass
        super().destroy()


def main():
    app = CandidateScanDebugGui()
    app.mainloop()


if __name__ == "__main__":
    main()
