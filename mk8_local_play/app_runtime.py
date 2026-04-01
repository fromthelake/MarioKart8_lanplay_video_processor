import json
import importlib.util
import os
import shutil
import subprocess
import sys
import cv2
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .project_paths import PROJECT_ROOT


@dataclass(frozen=True)
class AppConfig:
    execution_mode: str
    export_image_format: str
    easyocr_gpu_mode: str
    overlap_ocr_mode: str
    overlap_ocr_consumers: int
    ocr_workers: int
    score_analysis_workers: int
    parallel_video_score_workers: int
    pass1_scan_workers: int
    ocr_consensus_frames: int
    pass1_segment_overlap_frames: int
    pass1_min_segment_frames: int
    write_debug_csv: bool
    write_debug_score_images: bool
    write_debug_linking_excel: bool
    low_res_max_source_height: int
    low_res_character_roi_pad_x: int
    low_res_character_roi_pad_y: int
    low_res_character_template_width: int
    low_res_character_template_height: int
    low_res_character_offset_x: int
    low_res_character_offset_y: int
    low_res_row12_character_fallback_min_confidence: int
    low_res_row12_character_fallback_min_position_score: float
    ultra_low_res_row_min_stddev: float
    ultra_low_res_row_min_edge_density: float
    ultra_low_res_blob_match_min_score: float
    ultra_low_res_blob_match_min_margin: float


def _parse_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return default


def _parse_mode(value, default: str = "auto") -> str:
    candidate = str(value or default).strip().lower()
    if candidate in {"auto", "gpu", "cpu"}:
        return candidate
    return default


def _parse_overlap_ocr_mode(value, default: str = "auto") -> str:
    candidate = str(value or default).strip().lower()
    if candidate in {"auto", "video", "race"}:
        return candidate
    return default


def _legacy_bool_to_mode(value, default: str = "auto") -> str:
    if value is None:
        return default
    return "gpu" if _parse_bool(value, False) else "cpu"


def _parse_mode_with_legacy_bool(value, default: str = "auto") -> str:
    if isinstance(value, bool):
        return _legacy_bool_to_mode(value, default)
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
        return _legacy_bool_to_mode(text, default)
    return _parse_mode(text, default)


def _parse_export_image_format(value, default: str = "png") -> str:
    candidate = str(value or default).strip().lower().lstrip(".")
    if candidate in {"jpg", "jpeg"}:
        return "jpg"
    if candidate == "png":
        return "png"
    return default


def _parse_float(value, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(value))
    except (TypeError, ValueError):
        return default


def _load_json_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def update_app_config_values(overrides: dict, base_dir: Optional[Path] = None) -> None:
    base_dir = Path(base_dir or PROJECT_ROOT)
    config_path = base_dir / "config" / "app_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    json_config = _load_json_config(config_path)
    json_config.update(overrides)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(json_config, handle, indent=2)
        handle.write("\n")


def load_app_config(base_dir: Optional[Path] = None) -> AppConfig:
    base_dir = Path(base_dir or PROJECT_ROOT)
    config_path = base_dir / "config" / "app_config.json"
    json_config = _load_json_config(config_path)

    cpu_count = os.cpu_count() or 1
    default_ocr_workers = max(1, min(16, cpu_count))
    default_score_workers = 6 if cpu_count >= 16 else 4
    default_parallel_video_score_workers = 2 if cpu_count >= 16 else 1
    if cpu_count >= 24:
        default_pass1_workers = 4
    elif cpu_count >= 16:
        default_pass1_workers = 3
    elif cpu_count >= 8:
        default_pass1_workers = 2
    else:
        default_pass1_workers = 1

    execution_mode = _parse_mode(
        os.environ.get("MK8_EXECUTION_MODE", json_config.get("execution_mode")),
        "cpu",
    )
    export_image_format = _parse_export_image_format(
        os.environ.get("MK8_EXPORT_IMAGE_FORMAT", json_config.get("export_image_format")),
        "png",
    )
    easyocr_gpu_mode_source = os.environ.get("MK8_EASYOCR_GPU_MODE")
    if easyocr_gpu_mode_source is None:
        easyocr_gpu_mode_source = json_config.get("easyocr_gpu_mode")
    if easyocr_gpu_mode_source is None:
        easyocr_gpu_mode_source = os.environ.get("MK8_EASYOCR_GPU")
    if easyocr_gpu_mode_source is None:
        easyocr_gpu_mode_source = json_config.get("easyocr_gpu")
    easyocr_gpu_mode = _parse_mode_with_legacy_bool(easyocr_gpu_mode_source, "auto")
    overlap_ocr_mode = _parse_overlap_ocr_mode(
        os.environ.get("MK8_OVERLAP_OCR_MODE", json_config.get("overlap_ocr_mode")),
        "auto",
    )
    overlap_ocr_consumers = _parse_int(
        os.environ.get("MK8_OVERLAP_OCR_CONSUMERS", json_config.get("overlap_ocr_consumers")),
        2,
    )
    ocr_workers = _parse_int(
        os.environ.get("MK8_OCR_WORKERS", json_config.get("ocr_workers")),
        default_ocr_workers,
    )
    score_analysis_workers = _parse_int(
        os.environ.get("MK8_SCORE_ANALYSIS_WORKERS", json_config.get("score_analysis_workers")),
        default_score_workers,
    )
    parallel_video_score_workers = _parse_int(
        os.environ.get("MK8_PARALLEL_VIDEO_SCORE_WORKERS", json_config.get("parallel_video_score_workers")),
        default_parallel_video_score_workers,
    )
    pass1_scan_workers = _parse_int(
        os.environ.get("MK8_PASS1_SCAN_WORKERS", json_config.get("pass1_scan_workers")),
        default_pass1_workers,
    )
    ocr_consensus_frames = _parse_int(
        os.environ.get("MK8_OCR_CONSENSUS_FRAMES", json_config.get("ocr_consensus_frames")),
        7,
    )
    pass1_segment_overlap_frames = _parse_int(
        os.environ.get("MK8_PASS1_SEGMENT_OVERLAP_FRAMES", json_config.get("pass1_segment_overlap_frames")),
        70 * 30,
    )
    pass1_min_segment_frames = _parse_int(
        os.environ.get("MK8_PASS1_MIN_SEGMENT_FRAMES", json_config.get("pass1_min_segment_frames")),
        15 * 60 * 30,
    )
    write_debug_csv = _parse_bool(
        os.environ.get("MK8_WRITE_DEBUG_CSV", json_config.get("write_debug_csv")),
        True,
    )
    write_debug_score_images = _parse_bool(
        os.environ.get("MK8_WRITE_DEBUG_SCORE_IMAGES", json_config.get("write_debug_score_images")),
        True,
    )
    write_debug_linking_excel = _parse_bool(
        os.environ.get("MK8_WRITE_DEBUG_LINKING_EXCEL", json_config.get("write_debug_linking_excel")),
        True,
    )
    low_res_max_source_height = _parse_int(
        os.environ.get("MK8_LOW_RES_MAX_SOURCE_HEIGHT", json_config.get("low_res_max_source_height")),
        479,
    )
    low_res_character_roi_pad_x = _parse_int(
        os.environ.get("MK8_LOW_RES_CHARACTER_ROI_PAD_X", json_config.get("low_res_character_roi_pad_x")),
        4,
        minimum=0,
    )
    low_res_character_roi_pad_y = _parse_int(
        os.environ.get("MK8_LOW_RES_CHARACTER_ROI_PAD_Y", json_config.get("low_res_character_roi_pad_y")),
        4,
        minimum=0,
    )
    low_res_character_template_width = _parse_int(
        os.environ.get("MK8_LOW_RES_CHARACTER_TEMPLATE_WIDTH", json_config.get("low_res_character_template_width")),
        51,
        minimum=1,
    )
    low_res_character_template_height = _parse_int(
        os.environ.get("MK8_LOW_RES_CHARACTER_TEMPLATE_HEIGHT", json_config.get("low_res_character_template_height")),
        52,
        minimum=1,
    )
    low_res_character_offset_x = _parse_int(
        os.environ.get("MK8_LOW_RES_CHARACTER_OFFSET_X", json_config.get("low_res_character_offset_x")),
        4,
        minimum=0,
    )
    low_res_character_offset_y = _parse_int(
        os.environ.get("MK8_LOW_RES_CHARACTER_OFFSET_Y", json_config.get("low_res_character_offset_y")),
        5,
        minimum=0,
    )
    low_res_row12_character_fallback_min_confidence = _parse_int(
        os.environ.get(
            "MK8_LOW_RES_ROW12_CHARACTER_FALLBACK_MIN_CONFIDENCE",
            json_config.get("low_res_row12_character_fallback_min_confidence"),
        ),
        75,
        minimum=0,
    )
    low_res_row12_character_fallback_min_position_score = _parse_float(
        os.environ.get(
            "MK8_LOW_RES_ROW12_CHARACTER_FALLBACK_MIN_POSITION_SCORE",
            json_config.get("low_res_row12_character_fallback_min_position_score"),
        ),
        0.45,
        minimum=0.0,
    )
    ultra_low_res_row_min_stddev = _parse_float(
        os.environ.get("MK8_ULTRA_LOW_RES_ROW_MIN_STDDEV", json_config.get("ultra_low_res_row_min_stddev")),
        18.0,
        minimum=0.0,
    )
    ultra_low_res_row_min_edge_density = _parse_float(
        os.environ.get("MK8_ULTRA_LOW_RES_ROW_MIN_EDGE_DENSITY", json_config.get("ultra_low_res_row_min_edge_density")),
        0.035,
        minimum=0.0,
    )
    ultra_low_res_blob_match_min_score = _parse_float(
        os.environ.get("MK8_ULTRA_LOW_RES_BLOB_MATCH_MIN_SCORE", json_config.get("ultra_low_res_blob_match_min_score")),
        0.58,
        minimum=0.0,
    )
    ultra_low_res_blob_match_min_margin = _parse_float(
        os.environ.get("MK8_ULTRA_LOW_RES_BLOB_MATCH_MIN_MARGIN", json_config.get("ultra_low_res_blob_match_min_margin")),
        0.10,
        minimum=0.0,
    )

    return AppConfig(
        execution_mode=execution_mode,
        export_image_format=export_image_format,
        easyocr_gpu_mode=easyocr_gpu_mode,
        overlap_ocr_mode=overlap_ocr_mode,
        overlap_ocr_consumers=overlap_ocr_consumers,
        ocr_workers=ocr_workers,
        score_analysis_workers=score_analysis_workers,
        parallel_video_score_workers=parallel_video_score_workers,
        pass1_scan_workers=pass1_scan_workers,
        ocr_consensus_frames=ocr_consensus_frames,
        pass1_segment_overlap_frames=pass1_segment_overlap_frames,
        pass1_min_segment_frames=pass1_min_segment_frames,
        write_debug_csv=write_debug_csv,
        write_debug_score_images=write_debug_score_images,
        write_debug_linking_excel=write_debug_linking_excel,
        low_res_max_source_height=low_res_max_source_height,
        low_res_character_roi_pad_x=low_res_character_roi_pad_x,
        low_res_character_roi_pad_y=low_res_character_roi_pad_y,
        low_res_character_template_width=low_res_character_template_width,
        low_res_character_template_height=low_res_character_template_height,
        low_res_character_offset_x=low_res_character_offset_x,
        low_res_character_offset_y=low_res_character_offset_y,
        low_res_row12_character_fallback_min_confidence=low_res_row12_character_fallback_min_confidence,
        low_res_row12_character_fallback_min_position_score=low_res_row12_character_fallback_min_position_score,
        ultra_low_res_row_min_stddev=ultra_low_res_row_min_stddev,
        ultra_low_res_row_min_edge_density=ultra_low_res_row_min_edge_density,
        ultra_low_res_blob_match_min_score=ultra_low_res_blob_match_min_score,
        ultra_low_res_blob_match_min_margin=ultra_low_res_blob_match_min_margin,
    )


def find_executable(name: str, extra_candidates: Optional[List[str]] = None) -> Optional[str]:
    path = shutil.which(name)
    if path:
        return path
    for candidate in extra_candidates or []:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def detect_gpu_runtime(config: AppConfig) -> dict:
    cuda_module = getattr(cv2, "cuda", None) if "cv2" in sys.modules else None
    cuda_devices = 0
    if cuda_module is not None:
        try:
            cuda_devices = int(cuda_module.getCudaEnabledDeviceCount())
        except Exception:
            cuda_devices = 0
    gpu_available = cuda_devices > 0
    opencl_available = False
    opencl_in_use = False
    try:
        opencl_available = bool(cv2.ocl.haveOpenCL())
        if opencl_available:
            cv2.ocl.setUseOpenCL(True)
            opencl_in_use = bool(cv2.ocl.useOpenCL())
    except Exception:
        opencl_available = False
        opencl_in_use = False
    mode = _parse_mode(config.execution_mode, "cpu")
    backend = "cpu"
    reason = "CPU mode selected" if mode == "cpu" else "No GPU backend available"
    if mode == "cpu":
        enabled = False
    elif gpu_available:
        enabled = True
        backend = "cuda"
        reason = f"CUDA device(s) available: {cuda_devices}"
    else:
        enabled = False
        if mode == "gpu":
            if opencl_in_use:
                enabled = True
                backend = "opencl"
                reason = "CUDA unavailable; using OpenCL because GPU mode was requested explicitly"
            else:
                reason = "Requested GPU mode, but neither CUDA nor OpenCL was available"
        elif opencl_in_use:
            reason = "Auto mode selected; CUDA unavailable, and OpenCL auto-selection is disabled on this machine profile"
        elif opencl_available:
            reason = "OpenCL detected but not enabled by OpenCV"
    return {
        "available": gpu_available,
        "enabled": enabled,
        "device_count": cuda_devices,
        "mode": mode,
        "backend": backend,
        "opencl_available": opencl_available,
        "opencl_in_use": opencl_in_use,
        "reason": reason,
    }


def easyocr_gpu_enabled(config: AppConfig) -> bool:
    try:
        import torch
    except Exception:
        torch = None
    mode = _parse_mode(config.easyocr_gpu_mode, "auto")
    if mode == "cpu":
        return False
    if torch is None:
        return False
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False
    return cuda_available


def effective_overlap_ocr_mode(config: AppConfig) -> str:
    mode = _parse_overlap_ocr_mode(config.overlap_ocr_mode, "auto")
    if mode != "auto":
        return mode
    return "race" if easyocr_gpu_enabled(config) else "video"


def detect_easyocr_runtime(config: AppConfig) -> dict:
    mode = _parse_mode(config.easyocr_gpu_mode, "auto")
    try:
        import torch
    except Exception:
        torch = None
    cuda_available = False
    cuda_devices = 0
    if torch is not None:
        try:
            cuda_available = bool(torch.cuda.is_available())
            cuda_devices = int(torch.cuda.device_count()) if cuda_available else 0
        except Exception:
            cuda_available = False
            cuda_devices = 0
    enabled = mode != "cpu" and cuda_available
    if mode == "cpu":
        reason = "CPU mode selected"
    elif cuda_available:
        reason = f"CUDA device(s) available: {cuda_devices}"
    elif mode == "gpu":
        reason = "Requested GPU mode, but CUDA was not available to EasyOCR"
    else:
        reason = "Auto mode selected, but CUDA was not available to EasyOCR"
    return {
        "available": cuda_available,
        "enabled": enabled,
        "device_count": cuda_devices,
        "mode": mode,
        "backend": "cuda" if enabled else "cpu",
        "reason": reason,
    }


def check_runtime(config: AppConfig, require_ffmpeg: bool = False) -> List[str]:
    issues = []
    if importlib.util.find_spec("easyocr") is None:
        issues.append("EasyOCR is not installed in the local environment.")
    if require_ffmpeg and not find_executable("ffmpeg"):
        issues.append("FFmpeg was not found on PATH.")
    return issues


def open_path(path: Path) -> None:
    path = Path(path)
    if sys.platform.startswith("win"):
        os.startfile(str(path))
        return
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=True)
        return
    subprocess.run(["xdg-open", str(path)], check=True)
