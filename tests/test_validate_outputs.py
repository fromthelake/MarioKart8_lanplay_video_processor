import csv
import shutil
import unittest
import uuid
from pathlib import Path

from tools.validate_outputs import (
    align_rows_for_comparison,
    compare_file_sets,
    load_csv_rows,
    normalize_cell_for_compare,
)


TEST_TMP_ROOT = Path.cwd() / ".codex_tmp" / "unit_tests"


def _make_case_dir(case_name: str) -> Path:
    case_dir = TEST_TMP_ROOT / f"{case_name}_{uuid.uuid4().hex}"
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


class ValidateOutputsTests(unittest.TestCase):
    def test_normalize_cell_for_compare_normalizes_numeric_text(self):
        self.assertEqual(normalize_cell_for_compare("15.0"), "15")
        self.assertEqual(normalize_cell_for_compare("15.2500000000"), "15.25")
        self.assertEqual(normalize_cell_for_compare(" Blaasie "), "Blaasie")

    def test_align_rows_for_comparison_projects_shared_columns_in_order(self):
        baseline_rows = [
            ["Video", "Race", "Track"],
            ["Demo", "1", "Waluigi Stadium"],
        ]
        current_rows = [
            ["Track", "Video", "Race", "Extra"],
            ["Waluigi Stadium", "Demo", "1", "ignored"],
        ]

        aligned_baseline, aligned_current = align_rows_for_comparison(baseline_rows, current_rows)

        self.assertEqual(aligned_baseline, [["Video", "Race", "Track"], ["Demo", "1", "Waluigi Stadium"]])
        self.assertEqual(aligned_current, [["Video", "Race", "Track"], ["Demo", "1", "Waluigi Stadium"]])

    def test_compare_file_sets_reports_missing_unexpected_and_mismatched(self):
        temp_path = _make_case_dir("compare_file_sets")
        try:
            baseline_dir = temp_path / "baseline"
            current_dir = temp_path / "current"
            baseline_dir.mkdir(parents=True, exist_ok=True)
            current_dir.mkdir(parents=True, exist_ok=True)

            (baseline_dir / "same.png").write_bytes(b"same")
            (current_dir / "same.png").write_bytes(b"same")
            (baseline_dir / "missing.png").write_bytes(b"missing")
            (current_dir / "unexpected.png").write_bytes(b"unexpected")
            (baseline_dir / "mismatch.png").write_bytes(b"old")
            (current_dir / "mismatch.png").write_bytes(b"new")

            missing, unexpected, mismatched = compare_file_sets(baseline_dir, current_dir)

            self.assertEqual(missing, ["missing.png"])
            self.assertEqual(unexpected, ["unexpected.png"])
            self.assertEqual(mismatched, ["mismatch.png"])
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)

    def test_load_csv_rows_filters_by_race_class(self):
        temp_path = _make_case_dir("load_csv_rows")
        try:
            csv_path = temp_path / "Tournament_Results.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Video", "Race", "Track"])
                writer.writerow(["Demo_A", "1", "Track A"])
                writer.writerow(["Demo_B", "1", "Track B"])

            rows = load_csv_rows(csv_path, race_class="Demo_B")

            self.assertEqual(rows, [["Video", "Race", "Track"], ["Demo_B", "1", "Track B"]])
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)
