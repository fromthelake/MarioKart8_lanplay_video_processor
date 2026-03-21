import json
import os
import shutil
import unittest
import uuid
from pathlib import Path

from mk8_local_play.app_runtime import load_app_config


TEST_TMP_ROOT = Path.cwd() / ".codex_tmp" / "unit_tests"


def _make_case_dir(case_name: str) -> Path:
    case_dir = TEST_TMP_ROOT / f"{case_name}_{uuid.uuid4().hex}"
    (case_dir / "config").mkdir(parents=True, exist_ok=True)
    return case_dir


class AppRuntimeConfigTests(unittest.TestCase):
    def test_load_app_config_uses_global_export_image_format_setting(self):
        case_dir = _make_case_dir("app_config_export_format")
        try:
            config_path = case_dir / "config" / "app_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "execution_mode": "cpu",
                        "export_image_format": "jpeg",
                    }
                ),
                encoding="utf-8",
            )

            config = load_app_config(case_dir)

            self.assertEqual(config.export_image_format, "jpg")
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    def test_env_override_wins_for_export_image_format(self):
        case_dir = _make_case_dir("app_config_export_override")
        try:
            config_path = case_dir / "config" / "app_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "execution_mode": "cpu",
                        "export_image_format": "png",
                    }
                ),
                encoding="utf-8",
            )

            previous = os.environ.get("MK8_EXPORT_IMAGE_FORMAT")
            os.environ["MK8_EXPORT_IMAGE_FORMAT"] = "jpg"
            try:
                config = load_app_config(case_dir)
            finally:
                if previous is None:
                    os.environ.pop("MK8_EXPORT_IMAGE_FORMAT", None)
                else:
                    os.environ["MK8_EXPORT_IMAGE_FORMAT"] = previous

            self.assertEqual(config.export_image_format, "jpg")
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    def test_load_app_config_reads_easyocr_gpu_mode_and_legacy_env_override(self):
        case_dir = _make_case_dir("app_config_easyocr_gpu")
        try:
            config_path = case_dir / "config" / "app_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "execution_mode": "cpu",
                        "easyocr_gpu": False,
                    }
                ),
                encoding="utf-8",
            )

            previous = os.environ.get("MK8_EASYOCR_GPU")
            os.environ["MK8_EASYOCR_GPU"] = "1"
            try:
                config = load_app_config(case_dir)
            finally:
                if previous is None:
                    os.environ.pop("MK8_EASYOCR_GPU", None)
                else:
                    os.environ["MK8_EASYOCR_GPU"] = previous

            self.assertEqual(config.easyocr_gpu_mode, "gpu")
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    def test_load_app_config_reads_overlap_ocr_consumers(self):
        case_dir = _make_case_dir("app_config_overlap_ocr_consumers")
        try:
            config_path = case_dir / "config" / "app_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "execution_mode": "cpu",
                        "overlap_ocr_consumers": 2,
                    }
                ),
                encoding="utf-8",
            )

            config = load_app_config(case_dir)

            self.assertEqual(config.overlap_ocr_consumers, 2)
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    def test_load_app_config_reads_overlap_ocr_mode(self):
        case_dir = _make_case_dir("app_config_overlap_ocr_mode")
        try:
            config_path = case_dir / "config" / "app_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "execution_mode": "cpu",
                        "overlap_ocr_mode": "race",
                    }
                ),
                encoding="utf-8",
            )

            config = load_app_config(case_dir)

            self.assertEqual(config.overlap_ocr_mode, "race")
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    def test_load_app_config_defaults_overlap_ocr_to_auto_with_two_consumers(self):
        case_dir = _make_case_dir("app_config_overlap_ocr_defaults")
        try:
            config = load_app_config(case_dir)

            self.assertEqual(config.overlap_ocr_mode, "auto")
            self.assertEqual(config.overlap_ocr_consumers, 2)
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)
