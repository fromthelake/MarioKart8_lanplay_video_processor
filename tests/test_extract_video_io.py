import unittest

from mk8_local_play import extract_video_io


class ExtractVideoIoTests(unittest.TestCase):
    def test_update_repair_progress_state_resets_when_progress_advances(self):
        stalled_elapsed_s, progress_seconds = extract_video_io._update_repair_progress_state(
            12.0,
            10.0,
            8.0,
            2.0,
        )
        self.assertEqual(stalled_elapsed_s, 0.0)
        self.assertEqual(progress_seconds, 12.0)

    def test_update_repair_progress_state_accumulates_when_progress_is_flat(self):
        stalled_elapsed_s, progress_seconds = extract_video_io._update_repair_progress_state(
            10.1,
            10.0,
            8.0,
            2.0,
            epsilon_s=0.25,
        )
        self.assertEqual(stalled_elapsed_s, 10.0)
        self.assertEqual(progress_seconds, 10.0)
