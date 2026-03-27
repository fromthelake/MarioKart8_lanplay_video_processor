import tempfile
import unittest
from pathlib import Path

from mk8_local_play.extract_common import extract_exported_frame_number, find_score_bundle_race_context_paths
from mk8_local_play.score_layouts import (
    DEFAULT_SCORE_LAYOUT_ID,
    LAN1_SCORE_LAYOUT_ID,
    score_layout_id_from_filename,
    score_layout_tag_from_id,
)


class TestScoreLayoutTaggedPaths(unittest.TestCase):
    def test_extract_exported_frame_number_handles_layout_suffixes(self):
        self.assertEqual(extract_exported_frame_number("Race_001_F8261_2p"), 8261)
        self.assertEqual(extract_exported_frame_number("Race_001_F8261_1p"), 8261)
        self.assertEqual(extract_exported_frame_number("anchor_6096_2p"), 6096)
        self.assertEqual(extract_exported_frame_number("anchor_6096_1p"), 6096)

    def test_score_layout_id_from_filename_handles_short_tags(self):
        self.assertEqual(score_layout_id_from_filename("Race_001_F8261_2p.jpg"), DEFAULT_SCORE_LAYOUT_ID)
        self.assertEqual(score_layout_id_from_filename("Race_001_F8261_1p.jpg"), LAN1_SCORE_LAYOUT_ID)
        self.assertEqual(score_layout_id_from_filename("anchor_6096_2p.jpg"), DEFAULT_SCORE_LAYOUT_ID)
        self.assertEqual(score_layout_id_from_filename("anchor_6096_1p.jpg"), LAN1_SCORE_LAYOUT_ID)

    def test_score_layout_tag_from_id(self):
        self.assertEqual(score_layout_tag_from_id(DEFAULT_SCORE_LAYOUT_ID), "2p")
        self.assertEqual(score_layout_tag_from_id(LAN1_SCORE_LAYOUT_ID), "1p")

    def test_find_score_bundle_race_context_paths_sorts_tagged_frames_by_frame_number(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = Path(temp_dir) / "Video" / "Race_001" / "2RaceScore"
            bundle_dir.mkdir(parents=True)
            for name in (
                "Race_001_F8267_2p.jpg",
                "Race_001_F8261_2p.jpg",
                "Race_001_F8264_2p.jpg",
            ):
                (bundle_dir / name).touch()

            paths = find_score_bundle_race_context_paths(str(Path(temp_dir) / "Video"), 1, "2RaceScore")

            self.assertEqual([path.name for path in paths], [
                "Race_001_F8261_2p.jpg",
                "Race_001_F8264_2p.jpg",
                "Race_001_F8267_2p.jpg",
            ])


if __name__ == "__main__":
    unittest.main()
