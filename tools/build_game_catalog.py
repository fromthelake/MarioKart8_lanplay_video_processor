import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "database" / "firestore-export.json"
OUTPUT_PATHS = [
    PROJECT_ROOT / "reference_data" / "game_catalog.json",
    PROJECT_ROOT / "mk8_local_play_data" / "reference_data" / "game_catalog.json",
]


def load_firestore_export() -> dict:
    with SOURCE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_cups(raw_cups: dict) -> list[dict]:
    cups = []
    for raw_cup in raw_cups.values():
        cups.append(
            {
                "cupIndex": int(raw_cup["cupIndex"]),
                "legacyId": str(raw_cup["legacyId"]),
                "oldCupIndex": int(raw_cup["oldCupIndex"]),
                "slug": str(raw_cup["slug"]),
                "nameUK": str(raw_cup["nameUK"]),
                "nameUS": str(raw_cup["nameUS"]),
                "nameNL": str(raw_cup["nameDutch"]),
            }
        )
    return sorted(cups, key=lambda item: item["cupIndex"])


def build_tracks(raw_tracks: dict) -> list[dict]:
    tracks = []
    for raw_track in raw_tracks.values():
        tracks.append(
            {
                "trackIndex": int(raw_track["trackIndex"]),
                "legacyId": str(raw_track["legacyId"]),
                "oldTrackIndex": int(raw_track["oldtrackIndex"]),
                "cupIndex": int(raw_track["cupIndex"]),
                "slug": str(raw_track["slug"]),
                "lapCount": int(raw_track["lapCount"]),
                "nameUK": str(raw_track["nameUK"]),
                "nameUS": str(raw_track["nameUS"]),
                "nameNL": str(raw_track["nameDutch"]),
                "nameMkwrscom": str(raw_track["nameMkwrscom"]),
            }
        )
    return sorted(tracks, key=lambda item: item["trackIndex"])


def build_characters(raw_characters: dict) -> list[dict]:
    characters = []
    for raw_character in raw_characters.values():
        slug = str(raw_character["slug"])
        characters.append(
            {
                "characterIndex": int(raw_character["characterIndex"]),
                "legacyId": str(raw_character["legacyId"]),
                "rosterIndex": int(raw_character["rosterIndex"]),
                "mk8builderIndex": int(raw_character["mk8builderIndex"]),
                "slug": slug,
                "iconKey": slug,
                "nameUK": str(raw_character["nameUK"]),
                "nameUS": str(raw_character["nameUS"]),
                "nameNL": str(raw_character["nameNL"]),
            }
        )
    return sorted(characters, key=lambda item: item["characterIndex"])


def build_catalog(data: dict) -> dict:
    return {
        "version": 1,
        "source": {
            "derivedFrom": "database/firestore-export.json",
            "naming": "nameUK",
        },
        "cups": build_cups(data["cups"]),
        "tracks": build_tracks(data["tracks"]),
        "characters": build_characters(data["characters"]),
    }


def write_catalog(catalog: dict) -> None:
    output_text = json.dumps(catalog, indent=2, ensure_ascii=True) + "\n"
    for output_path in OUTPUT_PATHS:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")


def main() -> None:
    data = load_firestore_export()
    catalog = build_catalog(data)
    write_catalog(catalog)
    print(f"Wrote game catalog to {OUTPUT_PATHS[0]}")
    print(f"Wrote packaged game catalog to {OUTPUT_PATHS[1]}")


if __name__ == "__main__":
    main()
