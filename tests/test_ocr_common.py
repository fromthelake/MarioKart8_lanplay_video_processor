import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from mk8_local_play.ocr_common import load_consensus_frame_entries


class TestLoadConsensusFrameEntries(unittest.TestCase):
    def test_loads_sibling_consensus_frames_without_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = Path(temp_dir) / "Race_001" / "2RaceScore"
            bundle_dir.mkdir(parents=True)
            image = np.full((8, 8, 3), 127, dtype=np.uint8)
            anchor_path = bundle_dir / "anchor_101.jpg"
            self.assertTrue(cv2.imwrite(str(anchor_path), image))
            for frame_number in (100, 101, 102):
                frame_path = bundle_dir / f"frame_{frame_number}.jpg"
                self.assertTrue(cv2.imwrite(str(frame_path), image))

            entries = load_consensus_frame_entries(
                str(anchor_path),
                None,
                Path(temp_dir),
                consensus_size=7,
            )

            self.assertEqual([frame_number for frame_number, _frame in entries], [100, 101, 102])


if __name__ == "__main__":
    unittest.main()
