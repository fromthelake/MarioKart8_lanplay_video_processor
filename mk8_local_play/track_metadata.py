from pathlib import Path
from typing import List, Tuple

from .game_catalog import TrackMetadata, load_game_catalog


def load_track_metadata(base_dir: Path | None = None) -> List[TrackMetadata]:
    return load_game_catalog(base_dir).tracks


def load_track_tuples(base_dir: Path | None = None) -> List[Tuple[int, str, str, int, str]]:
    catalog = load_game_catalog(base_dir)
    cup_name_by_index = {cup.cup_index: cup.name_uk for cup in catalog.cups}
    return [track.as_legacy_tuple(cup_name_by_index.get(track.cup_index, "")) for track in catalog.tracks]
