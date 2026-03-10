import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from .data_paths import resolve_game_catalog_file


@dataclass(frozen=True)
class CupMetadata:
    cup_index: int
    legacy_id: str
    old_cup_index: int
    slug: str
    name_uk: str
    name_us: str
    name_nl: str


@dataclass(frozen=True)
class TrackMetadata:
    track_index: int
    legacy_id: str
    old_track_index: int
    cup_index: int
    slug: str
    lap_count: int
    name_uk: str
    name_us: str
    name_nl: str
    name_mkwrscom: str

    def as_legacy_tuple(self, cup_name_uk: str) -> Tuple[int, str, str, int, str]:
        return (self.track_index, self.name_uk, self.name_nl, self.cup_index, cup_name_uk)


@dataclass(frozen=True)
class CharacterMetadata:
    character_index: int
    legacy_id: str
    roster_index: int
    mk8builder_index: int
    slug: str
    icon_key: str
    name_uk: str
    name_us: str
    name_nl: str


@dataclass(frozen=True)
class GameCatalog:
    version: int
    cups: List[CupMetadata]
    tracks: List[TrackMetadata]
    characters: List[CharacterMetadata]


def load_game_catalog(base_dir: Path | None = None) -> GameCatalog:
    catalog_path = Path(base_dir) / "reference_data" / "game_catalog.json" if base_dir else resolve_game_catalog_file()
    with catalog_path.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    cups = [
        CupMetadata(
            cup_index=int(item["cupIndex"]),
            legacy_id=str(item["legacyId"]),
            old_cup_index=int(item["oldCupIndex"]),
            slug=str(item["slug"]),
            name_uk=str(item["nameUK"]),
            name_us=str(item["nameUS"]),
            name_nl=str(item["nameNL"]),
        )
        for item in raw_data["cups"]
    ]
    tracks = [
        TrackMetadata(
            track_index=int(item["trackIndex"]),
            legacy_id=str(item["legacyId"]),
            old_track_index=int(item["oldTrackIndex"]),
            cup_index=int(item["cupIndex"]),
            slug=str(item["slug"]),
            lap_count=int(item["lapCount"]),
            name_uk=str(item["nameUK"]),
            name_us=str(item["nameUS"]),
            name_nl=str(item["nameNL"]),
            name_mkwrscom=str(item["nameMkwrscom"]),
        )
        for item in raw_data["tracks"]
    ]
    characters = [
        CharacterMetadata(
            character_index=int(item["characterIndex"]),
            legacy_id=str(item["legacyId"]),
            roster_index=int(item["rosterIndex"]),
            mk8builder_index=int(item["mk8builderIndex"]),
            slug=str(item["slug"]),
            icon_key=str(item["iconKey"]),
            name_uk=str(item["nameUK"]),
            name_us=str(item["nameUS"]),
            name_nl=str(item["nameNL"]),
        )
        for item in raw_data["characters"]
    ]
    return GameCatalog(
        version=int(raw_data["version"]),
        cups=cups,
        tracks=tracks,
        characters=characters,
    )


def load_track_tuples(base_dir: Path | None = None) -> List[Tuple[int, str, str, int, str]]:
    catalog = load_game_catalog(base_dir)
    cup_name_by_index = {cup.cup_index: cup.name_uk for cup in catalog.cups}
    return [track.as_legacy_tuple(cup_name_by_index.get(track.cup_index, "")) for track in catalog.tracks]
