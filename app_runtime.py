import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class AppConfig:
    tesseract_cmd: Optional[str]
    ocr_workers: int
    score_analysis_workers: int
    pass1_scan_workers: int
    write_debug_csv: bool
    write_debug_score_images: bool
    write_debug_linking_excel: bool


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


def _load_json_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def load_app_config(base_dir: Optional[Path] = None) -> AppConfig:
    base_dir = Path(base_dir or Path(__file__).resolve().parent)
    config_path = base_dir / "app_config.json"
    json_config = _load_json_config(config_path)

    default_ocr_workers = max(1, min(16, os.cpu_count() or 1))
    default_score_workers = max(1, min(4, os.cpu_count() or 1))
    cpu_count = os.cpu_count() or 1
    if cpu_count >= 24:
        default_pass1_workers = 4
    elif cpu_count >= 16:
        default_pass1_workers = 3
    elif cpu_count >= 8:
        default_pass1_workers = 2
    else:
        default_pass1_workers = 1

    tesseract_cmd = os.environ.get("MK8_TESSERACT_CMD", json_config.get("tesseract_cmd"))
    ocr_workers = _parse_int(
        os.environ.get("MK8_OCR_WORKERS", json_config.get("ocr_workers")),
        default_ocr_workers,
    )
    score_analysis_workers = _parse_int(
        os.environ.get("MK8_SCORE_ANALYSIS_WORKERS", json_config.get("score_analysis_workers")),
        default_score_workers,
    )
    pass1_scan_workers = _parse_int(
        os.environ.get("MK8_PASS1_SCAN_WORKERS", json_config.get("pass1_scan_workers")),
        default_pass1_workers,
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

    return AppConfig(
        tesseract_cmd=tesseract_cmd,
        ocr_workers=ocr_workers,
        score_analysis_workers=score_analysis_workers,
        pass1_scan_workers=pass1_scan_workers,
        write_debug_csv=write_debug_csv,
        write_debug_score_images=write_debug_score_images,
        write_debug_linking_excel=write_debug_linking_excel,
    )


def _tesseract_candidates() -> List[str]:
    candidates = []
    if sys.platform.startswith("win"):
        candidates.extend(
            [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            ]
        )
    else:
        candidates.extend(
            [
                "/usr/bin/tesseract",
                "/usr/local/bin/tesseract",
                "/opt/homebrew/bin/tesseract",
            ]
        )
    return candidates


def find_executable(name: str, extra_candidates: Optional[List[str]] = None) -> Optional[str]:
    path = shutil.which(name)
    if path:
        return path
    for candidate in extra_candidates or []:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def resolve_tesseract_cmd(config: AppConfig) -> Optional[str]:
    if config.tesseract_cmd and Path(config.tesseract_cmd).exists():
        return config.tesseract_cmd
    return find_executable("tesseract", _tesseract_candidates())


def check_runtime(config: AppConfig, require_tesseract: bool = False, require_ffmpeg: bool = False) -> List[str]:
    issues = []
    if require_tesseract and not resolve_tesseract_cmd(config):
        issues.append(
            "Tesseract was not found. Install it or set MK8_TESSERACT_CMD / app_config.json:tesseract_cmd."
        )
    if require_ffmpeg and not find_executable("ffmpeg"):
        issues.append("FFmpeg was not found on PATH.")
    return issues


def configure_tesseract(pytesseract_module, config: AppConfig) -> str:
    tesseract_cmd = resolve_tesseract_cmd(config)
    if not tesseract_cmd:
        raise RuntimeError(
            "Tesseract was not found. Install it or set MK8_TESSERACT_CMD / app_config.json:tesseract_cmd."
        )
    pytesseract_module.pytesseract.tesseract_cmd = tesseract_cmd
    return tesseract_cmd


def open_path(path: Path) -> None:
    path = Path(path)
    if sys.platform.startswith("win"):
        os.startfile(str(path))
        return
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=True)
        return
    subprocess.run(["xdg-open", str(path)], check=True)
