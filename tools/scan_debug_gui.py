"""Legacy scanner debug GUI.

Superseded by tools/score_detail_debug_gui.py for detail-phase debugging.
Kept for backward compatibility with existing local workflows.
"""

import threading
import time
from collections import defaultdict
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from mk8_local_play.data_paths import resolve_asset_file
from mk8_local_play.extract_common import crop_and_upscale_image, frame_to_timecode
from mk8_local_play.extract_frames import _prepare_video_context, _run_scan_phase_for_context
import mk8_local_play.extract_initial_scan as initial_scan
import mk8_local_play.extract_score_screen_selection as score_sel
from mk8_local_play.extract_video_io import position_capture_for_read, read_video_frame, seek_to_frame
from mk8_local_play.ocr_scoreboard_consensus import extract_points_transition_observation, parse_detected_int
from mk8_local_play.project_paths import PROJECT_ROOT


class _NullWriter:
    def writerow(self, _row):
        return None

    def writerows(self, _rows):
        return None


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


def _simulate_transition_rows(observations):
    rows = []
    previous = None
    first_trigger = None
    for frame_number, obs in observations:
        changed_total_rows = 0
        changed_race_rows = 0
        changed_any_rows = 0
        if previous is not None:
            for row_index in range(6):
                prev_race = parse_detected_int(previous["race_points"][row_index])
                cur_race = parse_detected_int(obs["race_points"][row_index])
                prev_total = parse_detected_int(previous["total_points"][row_index])
                cur_total = parse_detected_int(obs["total_points"][row_index])
                race_changed = prev_race is not None and cur_race is not None and prev_race != cur_race
                total_changed = prev_total is not None and cur_total is not None and prev_total != cur_total
                changed_race_rows += int(race_changed)
                changed_total_rows += int(total_changed)
                changed_any_rows += int(race_changed or total_changed)
        triggered = (
            previous is not None
            and changed_total_rows >= 2
            and (changed_race_rows >= 1 or changed_any_rows >= 3)
        )
        if triggered and first_trigger is None:
            first_trigger = int(frame_number)
        rows.append(
            {
                "frame": int(frame_number),
                "checked_by_code": True,
                "triggered": triggered,
                "changed_race_rows": int(changed_race_rows),
                "changed_total_rows": int(changed_total_rows),
                "changed_any_rows": int(changed_any_rows),
                "race_points_top6": "|".join(str(x) for x in obs.get("race_points", [])[:6]),
                "total_points_top6": "|".join(str(x) for x in obs.get("total_points", [])[:6]),
            }
        )
        previous = obs
    return rows, first_trigger


def build_detail_trace(context, candidate, templates, progress_cb=None):
    def report(phase, percent):
        if progress_cb is None:
            return
        try:
            progress_cb(str(phase), float(percent))
        except Exception:
            pass

    fps = float(context["fps"])
    score_layout_id = str(candidate.get("score_layout_id") or "")
    candidate_frame = int(candidate["frame_number"])
    start_frame = candidate_frame - int(3 * fps)
    end_frame = candidate_frame + int(13 * fps)
    race_num = int(candidate["race_number"])

    cap = cv2.VideoCapture(context["processing_video_path"])
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {context['processing_video_path']}")

    try:
        report("Preparing detail scan", 0.0)
        stats = defaultdict(float)
        coarse_step = max(1, int(score_sel.COARSE_SEARCH_STEP_FRAMES))
        coarse_rewind = max(1, int(score_sel.COARSE_SEARCH_REWIND_FRAMES))
        detail_frame_number = int(start_frame)
        race_score_frame = 0
        score_hit_frame = None
        transition_frame = None
        selected_points_anchor_frame = None
        total_score_frame = None
        visited = {}
        transition_observations = []

        position_capture_for_read(
            cap,
            detail_frame_number,
            stats,
            max_forward_grab_frames=score_sel.SMALL_FORWARD_GRAB_WINDOW_FRAMES,
            label="gui_detail_start",
        )
        last_report = -1
        while detail_frame_number < int(end_frame):
            ret, frame = read_video_frame(cap, stats)
            if not ret:
                break
            if end_frame > start_frame:
                progress = ((int(detail_frame_number) - int(start_frame)) / float(int(end_frame) - int(start_frame))) * 65.0
                progress_int = int(progress)
                if progress_int != last_report and progress_int % 2 == 0:
                    last_report = progress_int
                    report("Scanning RaceScore detail frames", progress)

            upscaled_image = crop_and_upscale_image(
                frame,
                context["median_left"],
                context["median_top"],
                context["median_crop_width"],
                context["median_crop_height"],
                1280,
                720,
            )
            gray_image = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)

            ignore_label = ""
            ignore_max = 0.0
            ignore_threshold = 0.0
            ignore_rejected_as_blank = True
            if race_score_frame == 0:
                ignore_match = score_sel._match_ignore_frame_target_detail(gray_image, templates, stats)
                ignore_label = str(ignore_match.get("label", ""))
                ignore_max = float(ignore_match.get("max_val", 0.0))
                ignore_threshold = float(ignore_match.get("match_threshold", 0.0))
                ignore_rejected_as_blank = bool(ignore_match.get("rejected_as_blank", True))

            if race_score_frame == 0:
                raw_confirm_passed, raw_confirm_score = score_sel._raw_fixed_grid_prefix_confirm(
                    upscaled_image,
                    required_players=score_sel.POSITION_SCAN_MIN_PLAYERS,
                    score_layout_id=score_layout_id,
                    stats=stats,
                )
                max_val = float(raw_confirm_score)
                rejected_as_blank = not bool(raw_confirm_passed)
                detected_layout_id = score_layout_id
                gate_rows = []
                for row_number in range(1, int(score_sel.POSITION_SCAN_MIN_PLAYERS) + 1):
                    tile_gray = initial_scan._initial_scan_gate_tile_roi(
                        gray_image,
                        int(row_number),
                        score_layout_id=score_layout_id,
                    )
                    tile_size = int(initial_scan.INITIAL_SCAN_GATE_TEMPLATE_GRID_SIZE)
                    x1 = int(initial_scan.INITIAL_SCAN_GATE_TEMPLATE_GRID_X)
                    if str(score_layout_id or "").strip() == str(initial_scan.LAN1_SCORE_LAYOUT_ID):
                        x1 += int(initial_scan.SCORE_LAYOUT_SHIFT_X)
                    y1 = int(initial_scan.INITIAL_SCAN_GATE_TEMPLATE_GRID_Y) + ((int(row_number) - 1) * tile_size)
                    bbox = (int(x1), int(y1), int(tile_size), int(tile_size))
                    if tile_gray is None or getattr(tile_gray, "size", 0) == 0:
                        gate_rows.append(
                            {
                                "row": int(row_number),
                                "score": 0.0,
                                "passed": False,
                                "bbox": bbox,
                            }
                        )
                        continue
                    coeff, _variant_name = initial_scan._best_initial_scan_gate_score(tile_gray, int(row_number))
                    gate_rows.append(
                        {
                            "row": int(row_number),
                            "score": float(coeff),
                            "passed": bool(float(coeff) >= float(initial_scan.POSITION_SCAN_MIN_ROW_COEFF)),
                            "bbox": bbox,
                        }
                    )
            else:
                max_val, rejected_as_blank, detected_layout_id, _layout_metrics = score_sel._match_score_target_layouts(
                    upscaled_image,
                    templates,
                    stats,
                    return_layout_metrics=True,
                    preferred_layout_ids=[score_layout_id] if score_layout_id else None,
                    stats_scope="detail",
                )
                gate_rows = []

            visited[int(detail_frame_number)] = {
                "frame": int(detail_frame_number),
                "checked_by_code": True,
                "max_val": float(max_val),
                "rejected_as_blank": bool(rejected_as_blank),
                "ignore_label": ignore_label,
                "ignore_max": float(ignore_max),
                "ignore_threshold": float(ignore_threshold),
                "ignore_rejected_as_blank": bool(ignore_rejected_as_blank),
                "raw_confirm_mode": bool(race_score_frame == 0),
                "detected_layout_id": str(detected_layout_id or ""),
                "race_score_frame_locked": int(race_score_frame) if race_score_frame else None,
                "score_hit_frame": int(score_hit_frame) if score_hit_frame is not None else None,
                "transition_frame": int(transition_frame) if transition_frame is not None else None,
                "points_anchor_frame": int(selected_points_anchor_frame) if selected_points_anchor_frame is not None else None,
                "gate_rows": gate_rows,
            }

            if max_val > 0.3 and not rejected_as_blank and race_score_frame == 0:
                if coarse_step > 1:
                    coarse_step = 1
                    detail_frame_number = max(int(start_frame), int(detail_frame_number) - coarse_rewind)
                    seek_to_frame(cap, detail_frame_number, stats, label="gui_detail_rewind")
                    continue
                score_hit_frame = int(detail_frame_number)
                race_score_frame = score_hit_frame + int(0.7 * fps)
                transition_search_end = min(
                    int(end_frame),
                    score_hit_frame + max(1, int(round(score_sel.POINTS_TRANSITION_SEARCH_END_SECONDS * max(fps, 1.0)))),
                )
                transition_frame, selected_points_anchor_frame = score_sel._find_points_transition_frame(
                    cap,
                    int(race_score_frame),
                    int(transition_search_end),
                    context["median_left"],
                    context["median_top"],
                    context["median_crop_width"],
                    context["median_crop_height"],
                    score_layout_id,
                    stats,
                    fps,
                    int(context.get("source_height", 0) or 0),
                )
                if transition_frame is not None:
                    if selected_points_anchor_frame is None:
                        selected_points_anchor_frame = max(0, int(transition_frame) - 2)
                    total_score_frame = score_sel._find_total_score_stable_frame(
                        cap,
                        int(transition_frame),
                        fps,
                        context["median_left"],
                        context["median_top"],
                        context["median_crop_width"],
                        context["median_crop_height"],
                        score_layout_id,
                        stats,
                    )
                    break

            if rejected_as_blank and race_score_frame == 0 and coarse_step > 1:
                next_frame = min(int(end_frame), int(detail_frame_number) + coarse_step)
                if next_frame <= int(detail_frame_number):
                    break
                detail_frame_number = next_frame
                position_capture_for_read(
                    cap,
                    int(detail_frame_number),
                    stats,
                    max_forward_grab_frames=score_sel.SMALL_FORWARD_GRAB_WINDOW_FRAMES,
                    label="gui_detail_skip",
                )
                continue

            detail_frame_number += 1

        if race_score_frame and transition_frame is not None:
            report("Scanning transition frames", 66.0)
            transition_start = int(race_score_frame)
            transition_end = min(
                int(end_frame),
                int(transition_start + max(1, int(round(score_sel.POINTS_TRANSITION_SEARCH_END_SECONDS * max(fps, 1.0))))),
            )
            seek_to_frame(cap, int(transition_start), stats, label="gui_transition_probe")
            for frame_number in range(int(transition_start), int(transition_end) + 1):
                ret, frame = read_video_frame(cap, stats)
                if not ret:
                    break
                image = crop_and_upscale_image(
                    frame,
                    context["median_left"],
                    context["median_top"],
                    context["median_crop_width"],
                    context["median_crop_height"],
                    1280,
                    720,
                )
                obs = extract_points_transition_observation(image, score_layout_id=score_layout_id)
                transition_observations.append((int(frame_number), obs))

        transition_rows, transition_rule_frame = _simulate_transition_rows(transition_observations)
        report("Scanning TotalScore stable frames", 78.0)

        total_rows_map = {}
        if transition_frame is not None:
            total_start = int(transition_frame)
            total_end = int(transition_frame) + max(1, int(round(score_sel.TOTAL_SCORE_STABLE_SEARCH_SECONDS * max(fps, 1.0))))
            for frame_number in range(total_start, total_end + 1):
                seek_to_frame(cap, int(frame_number), stats, label="gui_total_probe")
                ret, frame = read_video_frame(cap, stats)
                if not ret:
                    break
                image = crop_and_upscale_image(
                    frame,
                    context["median_left"],
                    context["median_top"],
                    context["median_crop_width"],
                    context["median_crop_height"],
                    1280,
                    720,
                )
                row_signal = score_sel._tie_aware_score_signal_present(
                    image,
                    score_layout_id=score_layout_id,
                    min_players=score_sel.POSITION_SCAN_MIN_PLAYERS,
                )
                signature = score_sel._extract_total_score_stable_signature(image, score_layout_id=score_layout_id)
                total_rows_map[int(frame_number)] = {
                    "frame": int(frame_number),
                    "checked_by_code": True,
                    "row_signal": bool(row_signal),
                    "signature": "" if signature is None else "|".join(str(v) for v in signature),
                    "stable_target_frame": int(total_score_frame) if total_score_frame is not None else None,
                }
                if total_end > total_start:
                    pct = 78.0 + (((int(frame_number) - int(total_start)) / float(int(total_end) - int(total_start))) * 20.0)
                    report("Scanning TotalScore stable frames", pct)

        race_rows = []
        for frame_number in range(int(start_frame), int(end_frame) + 1):
            row = {
                "frame": int(frame_number),
                "checked_by_code": frame_number in visited,
            }
            if frame_number in visited:
                row.update(visited[frame_number])
            race_rows.append(row)

        transition_rows_map = {int(row["frame"]): row for row in transition_rows}
        if transition_rows_map:
            tr_start = min(transition_rows_map.keys())
            tr_end = max(transition_rows_map.keys())
            transition_view_rows = []
            for frame_number in range(tr_start, tr_end + 1):
                row = {"frame": int(frame_number), "checked_by_code": frame_number in transition_rows_map}
                if frame_number in transition_rows_map:
                    row.update(transition_rows_map[frame_number])
                transition_view_rows.append(row)
        else:
            transition_view_rows = []

        if total_rows_map:
            total_start = min(total_rows_map.keys())
            total_end = max(total_rows_map.keys())
            total_view_rows = []
            for frame_number in range(total_start, total_end + 1):
                row = {"frame": int(frame_number), "checked_by_code": frame_number in total_rows_map}
                if frame_number in total_rows_map:
                    row.update(total_rows_map[frame_number])
                total_view_rows.append(row)
        else:
            total_view_rows = []

        summary = {
            "race_number": int(race_num),
            "candidate_frame": int(candidate_frame),
            "score_hit_frame": score_hit_frame,
            "race_score_frame": race_score_frame if race_score_frame else None,
            "transition_frame": transition_frame,
            "points_anchor_frame": selected_points_anchor_frame,
            "transition_rule_frame": transition_rule_frame,
            "total_score_frame": total_score_frame,
        }
        report("Finalizing trace", 99.0)
        return race_rows, transition_view_rows, total_view_rows, summary
    finally:
        report("Detail trace ready", 100.0)
        cap.release()


class ScanDebugGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MK8 Scanner Debug GUI")
        self.geometry("1840x980")

        self.templates = _load_templates()
        self.video_path = None
        self.context = None
        self.score_candidates = []

        self.race_rows = []
        self.transition_rows = []
        self.total_rows = []
        self.summary = {}

        self.current_rows = []
        self.current_row_map = {}
        self.range_start_frame = 0
        self.range_end_frame = 0
        self.current_frame = 0
        self.current_mode = tk.StringVar(value="RaceScore Detail")
        self.is_playing = False
        self.is_production_playing = False
        self.production_frames = []
        self.production_index = 0
        self.production_play_btn_text = tk.StringVar(value="Play Production Code")
        self.pause_seconds_var = tk.StringVar(value="0.20")
        self.status_var = tk.StringVar(value="Select a video and run initial scan.")
        self.trace_progress_var = tk.StringVar(value="")
        self.trace_progress_value = tk.DoubleVar(value=0.0)
        self.code_parity_var = tk.StringVar(value="")
        self.summary_var = tk.StringVar(value="")
        self._trace_load_token = 0
        self.jump_frame_var = tk.StringVar(value="")
        self.production_checked_frames_all = set()
        self.production_checked_phase_map = {}
        self.production_frame_entries = []
        self._production_row_override = None
        self._production_phase_override = None
        self.trace_ready = False

        self.capture = None
        self.photo = None

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=8)

        ttk.Button(top, text="Select Video", command=self._on_select_video).pack(side=tk.LEFT)
        self.video_label = ttk.Label(top, text="No video selected", width=120)
        self.video_label.pack(side=tk.LEFT, padx=8)
        ttk.Button(top, text="Run Initial Scan", command=self._on_run_initial_scan).pack(side=tk.LEFT, padx=6)

        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        left = ttk.Frame(main, width=300)
        center = ttk.Frame(main)
        right = ttk.Frame(main, width=500)
        main.add(left, weight=1)
        main.add(center, weight=6)
        main.add(right, weight=2)

        ttk.Label(left, text="Detected RaceScore Candidates").pack(anchor=tk.W)
        self.candidate_list = tk.Listbox(left, height=25)
        self.candidate_list.pack(fill=tk.BOTH, expand=True, pady=4)
        self.candidate_list.bind("<<ListboxSelect>>", self._on_candidate_selected)
        ttk.Label(left, text="Mode").pack(anchor=tk.W, pady=(8, 2))
        mode_combo = ttk.Combobox(
            left,
            textvariable=self.current_mode,
            values=["RaceScore Detail", "Transition Scan", "TotalScore Scan"],
            state="readonly",
        )
        mode_combo.pack(fill=tk.X)
        mode_combo.bind("<<ComboboxSelected>>", self._on_mode_changed)

        self.video_canvas = tk.Label(center, bg="#111111")
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        self.video_canvas.bind("<Configure>", lambda _e: self._render_current_frame())

        controls_center = ttk.Frame(center)
        controls_center.pack(fill=tk.X, pady=6)
        row1 = ttk.Frame(controls_center)
        row1.pack(fill=tk.X, pady=2)
        ttk.Button(row1, text="Load Selected Race", command=self._load_selected_race).pack(side=tk.LEFT)
        ttk.Button(row1, text="Frame -1", command=lambda: self._step_frames(-1)).pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="Frame +1", command=lambda: self._step_frames(1)).pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="Jump -1s", command=lambda: self._step_seconds(-1.0)).pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="Jump +1s", command=lambda: self._step_seconds(1.0)).pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="Play/Pause", command=self._toggle_play).pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, textvariable=self.production_play_btn_text, command=self._toggle_play_production).pack(side=tk.LEFT, padx=4)

        row2 = ttk.Frame(controls_center)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="Jump to frame:").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.jump_frame_var, width=10).pack(side=tk.LEFT, padx=4)
        ttk.Button(row2, text="Go", command=self._jump_to_frame_from_entry).pack(side=tk.LEFT)
        ttk.Label(row2, text="Pause/frame (s):").pack(side=tk.LEFT, padx=(16, 4))
        ttk.Entry(row2, textvariable=self.pause_seconds_var, width=7).pack(side=tk.LEFT)

        ttk.Label(right, text="Summary").pack(anchor=tk.W)
        self.summary_text = tk.Text(right, height=11, wrap=tk.WORD)
        self.summary_text.pack(fill=tk.X, pady=(0, 6))
        self.summary_text.configure(state=tk.DISABLED)
        ttk.Label(right, text="Code Frame Parity").pack(anchor=tk.W)
        self.code_parity_label = ttk.Label(right, textvariable=self.code_parity_var, foreground="red")
        self.code_parity_label.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(right, text="Frame Metrics").pack(anchor=tk.W)
        self.metrics_text = tk.Text(right, height=45, wrap=tk.WORD)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        self.metrics_text.tag_config("bool_true", foreground="green")
        self.metrics_text.tag_config("bool_false", foreground="red")

        status = ttk.Label(self, textvariable=self.status_var)
        status.pack(fill=tk.X, padx=8, pady=(0, 6))
        trace_progress = ttk.Label(self, textvariable=self.trace_progress_var)
        trace_progress.pack(fill=tk.X, padx=8, pady=(0, 4))
        self.progress = ttk.Progressbar(
            self,
            mode="determinate",
            maximum=100.0,
            variable=self.trace_progress_value,
        )
        self.progress.pack(fill=tk.X, padx=8, pady=(0, 6))
        self.trace_progress_value.set(0.0)

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
        self.status_var.set("Video selected. Running initial scan...")
        self._on_run_initial_scan()

    def _on_run_initial_scan(self):
        if self.video_path is None:
            messagebox.showwarning("No video", "Select a video first.")
            return

        def worker():
            try:
                self.status_var.set("Running initial scan...")
                folder_path = str(PROJECT_ROOT / "Input_Videos")
                include_subfolders = True
                context = _prepare_video_context(
                    str(self.video_path),
                    folder_path,
                    include_subfolders,
                    1,
                    1,
                    0.0,
                    self.templates,
                    video_label=self.video_path.stem,
                    source_display_name=str(self.video_path),
                )
                if context is None:
                    raise RuntimeError("Failed to prepare video context")
                scan_result = _run_scan_phase_for_context(context, self.templates, _NullWriter(), _NullWriter())
                candidates = list(scan_result.get("score_candidates") or [])
                candidates.sort(key=lambda item: int(item.get("race_number", 0)))
                self.after(0, lambda: self._on_scan_complete(context, candidates))
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("Scan failed", str(exc)))
                self.after(0, lambda: self.status_var.set("Initial scan failed."))

        threading.Thread(target=worker, daemon=True).start()

    def _on_scan_complete(self, context, candidates):
        self.context = context
        self.score_candidates = candidates
        self.candidate_list.delete(0, tk.END)
        for candidate in candidates:
            race = int(candidate.get("race_number", 0))
            frame = int(candidate.get("frame_number", 0))
            label = f"Race {race:03d} | Candidate frame {frame} | {frame_to_timecode(frame, float(context['fps']))}"
            self.candidate_list.insert(tk.END, label)
        self.candidate_list.selection_clear(0, tk.END)
        self.status_var.set(f"Initial scan done. Found {len(candidates)} race candidates. Select a race to load detail scan.")

    def _on_candidate_selected(self, _event=None):
        index = self.candidate_list.curselection()
        if not index:
            return
        race_idx = int(index[0])
        if race_idx < len(self.score_candidates):
            candidate = self.score_candidates[race_idx]
            race_number = int(candidate.get("race_number", 0))
            self.status_var.set(f"Selected race {race_number:03d}. Loading trace...")
            self._reset_playback_state()
            self._show_race_preview(candidate)
            self._load_selected_race()

    def _show_race_preview(self, candidate):
        if self.context is None:
            return
        fps = float(self.context["fps"])
        candidate_frame = int(candidate.get("frame_number", 0))
        start_frame = max(0, int(candidate_frame - int(3 * fps)))
        end_frame = max(start_frame, int(candidate_frame + int(13 * fps)))
        self.current_rows = []
        self.current_row_map = {}
        self.trace_ready = False
        self._production_row_override = None
        self._production_phase_override = None
        self.range_start_frame = int(start_frame)
        self.range_end_frame = int(end_frame)
        self.current_frame = int(start_frame)
        self._set_summary_lines(
            [
                ("Race", f"{int(candidate.get('race_number', 0)):03d}", None),
                ("Candidate Frame", str(candidate_frame), int(candidate_frame)),
                ("Detail Range Start", f"{start_frame} (-3.0s)", int(start_frame)),
                ("Detail Range End", f"{end_frame} (+13.0s)", int(end_frame)),
            ]
        )
        self._render_current_frame()

    def _load_selected_race(self):
        if self.context is None or not self.score_candidates:
            messagebox.showwarning("No scan data", "Run initial scan first.")
            return
        selected = self.candidate_list.curselection()
        if not selected:
            messagebox.showwarning("No race selected", "Select a race from the list.")
            return
        candidate = self.score_candidates[int(selected[0])]
        self._reset_playback_state()
        self._trace_load_token += 1
        load_token = int(self._trace_load_token)
        self._set_trace_progress("starting", 0.0)

        def worker():
            try:
                race_num = int(candidate.get("race_number", 0))
                self.after(0, lambda: self.status_var.set(f"Building detail trace for race {race_num:03d}..."))
                def progress_update(phase, percent):
                    self.after(0, lambda: self._set_trace_progress(phase, percent))

                race_rows, transition_rows, total_rows, summary = build_detail_trace(
                    self.context,
                    candidate,
                    self.templates,
                    progress_cb=progress_update,
                )
                self.after(0, lambda: self._on_trace_loaded(load_token, race_rows, transition_rows, total_rows, summary))
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("Trace failed", str(exc)))
                self.after(0, lambda: self.status_var.set("Detail trace failed."))
                self.after(0, lambda: self._set_trace_progress("failed", 0.0))

        threading.Thread(target=worker, daemon=True).start()

    def _on_trace_loaded(self, load_token, race_rows, transition_rows, total_rows, summary):
        if int(load_token) != int(self._trace_load_token):
            return
        self.race_rows = race_rows
        self.transition_rows = transition_rows
        self.total_rows = total_rows
        self.summary = summary
        self._rebuild_production_checked_maps()
        self.trace_ready = True
        self._production_row_override = None
        self._production_phase_override = None
        self._set_mode_rows()
        fps = max(1.0, float(self.context["fps"]))
        candidate_frame = summary.get("candidate_frame")
        range_start = candidate_frame - int(3 * fps) if candidate_frame is not None else None
        range_end = candidate_frame + int(13 * fps) if candidate_frame is not None else None
        selected_frames = []
        for value in (
            summary.get("candidate_frame"),
            range_start,
            range_end,
            summary.get("score_hit_frame"),
            summary.get("race_score_frame"),
            summary.get("transition_frame"),
            summary.get("points_anchor_frame"),
            summary.get("transition_rule_frame"),
            summary.get("total_score_frame"),
        ):
            if value is None:
                continue
            frame_value = int(value)
            if frame_value not in selected_frames:
                selected_frames.append(frame_value)
        self._set_summary_lines(
            [
                ("Race", f"{int(summary.get('race_number', 0)):03d}", None),
                ("Candidate Frame", str(summary.get("candidate_frame")), summary.get("candidate_frame")),
                ("Detail Range Start", f"{range_start} (-3.0s)", range_start),
                ("Detail Range End", f"{range_end} (+13.0s)", range_end),
                ("Score Hit Frame", str(summary.get("score_hit_frame")), summary.get("score_hit_frame")),
                ("Race Anchor Frame", str(summary.get("race_score_frame")), summary.get("race_score_frame")),
                ("Transition Frame", str(summary.get("transition_frame")), summary.get("transition_frame")),
                ("Points Anchor Frame", str(summary.get("points_anchor_frame")), summary.get("points_anchor_frame")),
                ("Transition Rule Frame", str(summary.get("transition_rule_frame")), summary.get("transition_rule_frame")),
                ("Total Anchor Frame", str(summary.get("total_score_frame")), summary.get("total_score_frame")),
                ("Selected Frames", ", ".join(str(v) for v in selected_frames), selected_frames),
            ]
        )
        self.status_var.set("Detail trace loaded.")
        self._set_trace_progress("ready", 100.0)

    def _set_trace_progress(self, phase, percent):
        clamped = max(0.0, min(100.0, float(percent)))
        self.trace_progress_var.set(f"Detail scan: {phase} ({clamped:.0f}%)")
        self.trace_progress_value.set(clamped)

    def _on_mode_changed(self, _event=None):
        self._set_mode_rows()

    def _set_mode_rows(self):
        mode = self.current_mode.get()
        if mode == "Transition Scan":
            self.current_rows = list(self.transition_rows)
        elif mode == "TotalScore Scan":
            self.current_rows = list(self.total_rows)
        else:
            self.current_rows = list(self.race_rows)
        self.current_row_map = {int(row.get("frame", -1)): row for row in self.current_rows if row.get("frame") is not None}
        if self.current_rows:
            self.range_start_frame = int(min(self.current_row_map.keys()))
            self.range_end_frame = int(max(self.current_row_map.keys()))
            self.current_frame = int(self.range_start_frame)
            self.jump_frame_var.set(str(self.current_frame))
        self._render_current_frame()

    def _rebuild_production_checked_maps(self):
        self.production_checked_frames_all = set()
        self.production_checked_phase_map = {}
        self.production_frame_entries = []
        for phase_name, rows in (
            ("RaceScore Detail", self.race_rows),
            ("Transition Scan", self.transition_rows),
            ("TotalScore Scan", self.total_rows),
        ):
            for row in rows:
                frame = int(row.get("frame", -1))
                if frame < 0:
                    continue
                if bool(row.get("checked_by_code", False)):
                    self.production_checked_frames_all.add(frame)
                    phases = self.production_checked_phase_map.setdefault(frame, set())
                    phases.add(phase_name)
                    self.production_frame_entries.append(
                        {
                            "frame": int(frame),
                            "phase": str(phase_name),
                            "row": dict(row),
                        }
                    )

    def _ensure_capture(self):
        if self.capture is not None:
            return
        if self.context is None:
            return
        self.capture = cv2.VideoCapture(self.context["processing_video_path"])

    def _render_current_frame(self):
        if self.range_end_frame < self.range_start_frame:
            self.code_parity_var.set("")
            self.metrics_text.delete("1.0", tk.END)
            return
        self.current_frame = max(int(self.range_start_frame), min(int(self.current_frame), int(self.range_end_frame)))
        frame_number = int(self.current_frame)
        if (
            self._production_row_override is not None
            and int(self._production_row_override.get("frame", -1)) == int(frame_number)
        ):
            row = dict(self._production_row_override)
            display_mode = str(self._production_phase_override or self.current_mode.get())
        else:
            row = self.current_row_map.get(frame_number, {"frame": frame_number, "checked_by_code": False})
            display_mode = self.current_mode.get()
        checked_by_code = bool(row.get("checked_by_code", False))

        self._ensure_capture()
        if self.capture is None or not self.capture.isOpened():
            return
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.capture.read()
        if ret and frame is not None:
            image = crop_and_upscale_image(
                frame,
                self.context["median_left"],
                self.context["median_top"],
                self.context["median_crop_width"],
                self.context["median_crop_height"],
                1280,
                720,
            )
            self._draw_roi_overlays(image, row)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            canvas_w = max(1, int(self.video_canvas.winfo_width()))
            canvas_h = max(1, int(self.video_canvas.winfo_height()))
            scale = min(float(canvas_w) / float(pil.width), float(canvas_h) / float(pil.height))
            scale = max(0.05, min(2.0, scale))
            resized_w = max(1, int(round(pil.width * scale)))
            resized_h = max(1, int(round(pil.height * scale)))
            pil = pil.resize((resized_w, resized_h), Image.Resampling.BILINEAR)
            self.photo = ImageTk.PhotoImage(pil)
            self.video_canvas.configure(image=self.photo)

        checked_any_phase = frame_number in self.production_checked_frames_all
        checked_phases = sorted(self.production_checked_phase_map.get(frame_number, set()))
        if checked_any_phase:
            phase_text = ", ".join(checked_phases) if checked_phases else "unknown phase"
            self.code_parity_var.set(f"Frame {frame_number} is checked by production code ({phase_text}).")
            self.code_parity_label.configure(foreground="green")
        else:
            self.code_parity_var.set(f"Frame {frame_number} is NOT checked by production code.")
            self.code_parity_label.configure(foreground="red")

        self.metrics_text.delete("1.0", tk.END)
        total_span = int(self.range_end_frame - self.range_start_frame + 1)
        frame_offset = int(frame_number - self.range_start_frame + 1)
        lines = [
            f"Mode: {display_mode}",
            f"Frame {frame_offset}/{total_span}",
            f"Frame: {frame_number} ({frame_to_timecode(frame_number, float(self.context['fps']))})",
        ]
        lines.extend(self._friendly_metrics_lines(row, display_mode=display_mode))
        self.metrics_text.insert("1.0", "\n".join(lines))
        self._colorize_metric_bools()
        self.jump_frame_var.set(str(frame_number))

    def _step_frames(self, delta):
        if self.range_end_frame < self.range_start_frame:
            return
        self.is_playing = False
        self.is_production_playing = False
        self.production_play_btn_text.set("Play Production Code")
        self._production_row_override = None
        self._production_phase_override = None
        self.current_frame = int(self.current_frame) + int(delta)
        self._render_current_frame()

    def _step_seconds(self, seconds):
        if self.context is None:
            return
        fps = max(1.0, float(self.context["fps"]))
        delta = int(round(float(seconds) * fps))
        self._step_frames(delta)

    def _toggle_play(self):
        self.is_production_playing = False
        self.production_play_btn_text.set("Play Production Code")
        self._production_row_override = None
        self._production_phase_override = None
        self.is_playing = not self.is_playing
        if self.is_playing:
            self._play_loop()

    def _play_loop(self):
        if not self.is_playing:
            return
        if self.range_end_frame < self.range_start_frame:
            self.is_playing = False
            return
        self._render_current_frame()
        if int(self.current_frame) >= int(self.range_end_frame):
            self.is_playing = False
            return
        self.current_frame += 1
        pause_seconds = _parse_pause_seconds(self.pause_seconds_var.get())
        delay_ms = max(1, int(round(pause_seconds * 1000.0)))
        self.after(delay_ms, self._play_loop)

    def _toggle_play_production(self):
        if not self.trace_ready:
            self.status_var.set("Trace is still loading. Wait for 'Detail trace loaded'.")
            return
        if self.range_end_frame < self.range_start_frame:
            return
        if not self.production_frame_entries:
            self.status_var.set("No production-checked frames in this mode range.")
            return
        self.is_playing = False
        self.is_production_playing = not self.is_production_playing
        self.production_play_btn_text.set("Pause Production Code" if self.is_production_playing else "Play Production Code")
        self.production_frames = list(self.production_frame_entries)
        try:
            next_idx = next(
                idx
                for idx, entry in enumerate(self.production_frames)
                if int(entry.get("frame", -1)) >= int(self.current_frame)
            )
        except StopIteration:
            next_idx = 0
        self.production_index = int(next_idx)
        if self.is_production_playing:
            self._play_production_loop()

    def _play_production_loop(self):
        if not self.is_production_playing:
            return
        if not self.production_frames or self.production_index >= len(self.production_frames):
            self.is_production_playing = False
            self.production_play_btn_text.set("Play Production Code")
            self._production_row_override = None
            self._production_phase_override = None
            return
        entry = self.production_frames[self.production_index]
        self.current_frame = int(entry.get("frame", 0))
        self._production_row_override = dict(entry.get("row") or {})
        self._production_phase_override = str(entry.get("phase") or "")
        self._render_current_frame()
        self.production_index += 1
        if self.production_index >= len(self.production_frames):
            self.is_production_playing = False
            self.production_play_btn_text.set("Play Production Code")
            self._production_row_override = None
            self._production_phase_override = None
            return
        pause_seconds = _parse_pause_seconds(self.pause_seconds_var.get())
        delay_ms = max(1, int(round(pause_seconds * 1000.0)))
        self.after(delay_ms, self._play_production_loop)

    def _jump_to_frame_from_entry(self):
        if self.range_end_frame < self.range_start_frame:
            return
        try:
            target = int(str(self.jump_frame_var.get()).strip())
        except ValueError:
            return
        self.is_playing = False
        self.is_production_playing = False
        self.production_play_btn_text.set("Play Production Code")
        self._production_row_override = None
        self._production_phase_override = None
        self.current_frame = int(target)
        self._render_current_frame()

    def _set_summary_lines(self, lines):
        self.summary_text.configure(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)
        for idx, (label, display_value, jump_frame) in enumerate(lines):
            self.summary_text.insert(tk.END, f"{label}: ")
            start_line = self.summary_text.index(tk.END)
            self.summary_text.insert(tk.END, f"{display_value}\n")
            if isinstance(jump_frame, list):
                numbers = [int(v) for v in jump_frame]
                line_text = str(display_value)
                for item_index, frame in enumerate(numbers):
                    needle = str(frame)
                    pos = line_text.find(needle)
                    if pos < 0:
                        continue
                    tag = f"jump_{idx}_{item_index}"
                    tag_start = f"{start_line}+{pos}c"
                    tag_end = f"{start_line}+{pos + len(needle)}c"
                    self.summary_text.tag_add(tag, tag_start, tag_end)
                    self.summary_text.tag_config(tag, foreground="blue", underline=1)
                    self.summary_text.tag_bind(tag, "<Button-1>", lambda _e, frame=int(frame): self._jump_to_summary_frame(frame))
            elif jump_frame is not None and str(display_value) not in {"None", ""}:
                tag = f"jump_{idx}"
                tag_start = start_line
                tag_end = f"{start_line}+{len(str(display_value))}c"
                self.summary_text.tag_add(tag, tag_start, tag_end)
                self.summary_text.tag_config(tag, foreground="blue", underline=1)
                self.summary_text.tag_bind(tag, "<Button-1>", lambda _e, frame=int(jump_frame): self._jump_to_summary_frame(frame))
        self.summary_text.configure(state=tk.DISABLED)

    def _jump_to_summary_frame(self, frame_number):
        self.is_playing = False
        self.is_production_playing = False
        self.production_play_btn_text.set("Play Production Code")
        self._production_row_override = None
        self._production_phase_override = None
        self.current_frame = int(frame_number)
        self._render_current_frame()

    def _friendly_metrics_lines(self, row, *, display_mode=None):
        lines = []
        checked = bool(row.get("checked_by_code", False))
        mode = str(display_mode or self.current_mode.get())
        lines.append("")
        lines.append("Goal")
        if mode == "RaceScore Detail":
            lines.append("- Find a stable RaceScore screen with enough position rows to lock score hit.")
        elif mode == "Transition Scan":
            lines.append("- Find the points animation transition (totals changing across rows).")
        else:
            lines.append("- Find a stable TotalScore frame after transition.")
        lines.append("")
        lines.append("Production Step")
        lines.append(f"- Production checked this frame: {'TRUE' if checked else 'FALSE'}")
        if mode == "RaceScore Detail":
            rejected = bool(row.get("rejected_as_blank", False))
            max_val = float(row.get("max_val", 0.0) or 0.0)
            score_trigger = bool(max_val > 0.30 and not rejected)
            gate_rows = row.get("gate_rows") or []
            passing_rows = sum(1 for gate in gate_rows if bool(gate.get("passed", False)))
            required_rows = int(score_sel.POSITION_SCAN_MIN_PLAYERS)
            lines.append(f"- Score-screen signal strong enough: {'TRUE' if score_trigger else 'FALSE'} (threshold > 0.30)")
            lines.append(f"- Rejected as blank: {'TRUE' if rejected else 'FALSE'}")
            lines.append(f"- Ignore overlay label: {row.get('ignore_label') or 'None'}")
            lines.append(f"- Ignore match score: {float(row.get('ignore_max', 0.0) or 0.0):.4f}")
            lines.append(f"- Ignore threshold: {float(row.get('ignore_threshold', 0.0) or 0.0):.2f}")
            lines.append(f"- Raw prefix mode active: {'TRUE' if bool(row.get('raw_confirm_mode', False)) else 'FALSE'}")
            lines.append(f"- Detected layout: {row.get('detected_layout_id') or 'unknown'}")
            lines.append(f"- Score coefficient: {max_val:.4f}")
            lines.append(f"- Position rows passing gate: {passing_rows}/{required_rows}")
            lines.append(f"- Score hit frame: {row.get('score_hit_frame')}")
            lines.append(f"- Race anchor frame: {row.get('race_score_frame_locked')}")
            lines.append(f"- Transition frame: {row.get('transition_frame')}")
            lines.append(f"- Points anchor frame: {row.get('points_anchor_frame')}")
            if gate_rows:
                lines.append("")
                lines.append("ROI Checks (Position Tiles)")
                for gate in gate_rows:
                    lines.append(
                        "- Row {row}: score={score:.4f} pass={passed} bbox={bbox}".format(
                            row=int(gate.get("row", 0)),
                            score=float(gate.get("score", 0.0)),
                            passed=("TRUE" if bool(gate.get("passed", False)) else "FALSE"),
                            bbox=gate.get("bbox"),
                        )
                    )
                lines.append("- Trigger logic: if score signal is TRUE, code locks Score Hit and starts Transition scan.")
        elif mode == "Transition Scan":
            lines.append(f"- Transition rule triggered: {'TRUE' if bool(row.get('triggered', False)) else 'FALSE'}")
            lines.append(f"- Changed race-point rows: {int(row.get('changed_race_rows', 0) or 0)}")
            lines.append(f"- Changed total-point rows: {int(row.get('changed_total_rows', 0) or 0)}")
            lines.append(f"- Changed rows (any): {int(row.get('changed_any_rows', 0) or 0)}")
            lines.append(f"- Race points top-6: {row.get('race_points_top6', '')}")
            lines.append(f"- Total points top-6: {row.get('total_points_top6', '')}")
            lines.append("- Trigger logic: totals changed on >=2 rows and (race changed >=1 row OR any changed >=3 rows).")
        else:
            lines.append(f"- Row signal present: {'TRUE' if bool(row.get('row_signal', False)) else 'FALSE'}")
            lines.append(f"- Stable total signature: {row.get('signature') or 'None'}")
            lines.append(f"- Stable target frame chosen by code: {row.get('stable_target_frame')}")
            lines.append("- Trigger logic: stable frame signature and row signal pass; then Total anchor is locked.")

        lines.append("")
        lines.append("Technical Details")
        for key in sorted(row.keys()):
            if key in {"frame", "gate_rows"}:
                continue
            lines.append(f"- {key}: {row.get(key)}")
        return lines

    def _reset_playback_state(self):
        self.is_playing = False
        self.is_production_playing = False
        self.production_frames = []
        self.production_index = 0
        self.production_play_btn_text.set("Play Production Code")
        self._production_row_override = None
        self._production_phase_override = None

    def _colorize_metric_bools(self):
        self.metrics_text.tag_remove("bool_true", "1.0", tk.END)
        self.metrics_text.tag_remove("bool_false", "1.0", tk.END)
        start = "1.0"
        while True:
            idx = self.metrics_text.search("TRUE", start, stopindex=tk.END)
            if not idx:
                break
            end = f"{idx}+4c"
            self.metrics_text.tag_add("bool_true", idx, end)
            start = end
        start = "1.0"
        while True:
            idx = self.metrics_text.search("FALSE", start, stopindex=tk.END)
            if not idx:
                break
            end = f"{idx}+5c"
            self.metrics_text.tag_add("bool_false", idx, end)
            start = end

    def _draw_roi_overlays(self, image, row):
        mode = str(self._production_phase_override or self.current_mode.get())
        if mode != "RaceScore Detail":
            return
        gate_rows = row.get("gate_rows") or []
        for gate in gate_rows:
            bbox = gate.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x, y, w, h = [int(v) for v in bbox]
            passed = bool(gate.get("passed", False))
            color = (0, 220, 0) if passed else (0, 140, 255)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
            cv2.putText(
                image,
                f"R{int(gate.get('row', 0))}:{float(gate.get('score', 0.0)):.2f}",
                (x + 2, max(12, y - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )
        ignore_label = str(row.get("ignore_label") or "").strip()
        if ignore_label:
            for target in initial_scan.IGNORE_FRAME_TARGETS:
                if str(target.get("label", "")) != ignore_label:
                    continue
                x, y, w, h = [int(v) for v in target.get("roi", (0, 0, 0, 0))]
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(
                    image,
                    f"{ignore_label}:{float(row.get('ignore_max', 0.0) or 0.0):.2f}",
                    (x + 2, max(12, y - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                break

    def destroy(self):
        self.is_playing = False
        self.is_production_playing = False
        self.production_play_btn_text.set("Play Production Code")
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        super().destroy()


def main():
    app = ScanDebugGui()
    app.mainloop()


if __name__ == "__main__":
    main()
