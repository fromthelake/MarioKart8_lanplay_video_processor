import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from .data_paths import resolve_track_metadata_file


@dataclass(frozen=True)
class TrackMetadata:
    track_id: int
    english_name: str
    dutch_name: str
    cup_order: int
    cup_name: str

    def as_legacy_tuple(self) -> Tuple[int, str, str, int, str]:
        return (self.track_id, self.english_name, self.dutch_name, self.cup_order, self.cup_name)


def load_track_metadata(base_dir: Path | None = None) -> List[TrackMetadata]:
    metadata_path = Path(base_dir) / "reference_data" / "track_metadata.json" if base_dir else resolve_track_metadata_file()
    with metadata_path.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)
    return [
        TrackMetadata(
            track_id=int(item["track_id"]),
            english_name=str(item["english_name"]),
            dutch_name=str(item["dutch_name"]),
            cup_order=int(item["cup_order"]),
            cup_name=str(item["cup_name"]),
        )
        for item in raw_data
    ]


def load_track_tuples(base_dir: Path | None = None) -> List[Tuple[int, str, str, int, str]]:
    return [track.as_legacy_tuple() for track in load_track_metadata(base_dir)]
