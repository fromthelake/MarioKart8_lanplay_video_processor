from importlib.resources import files
from pathlib import Path

from .project_paths import PROJECT_ROOT


def _package_root():
    return files("mk8_local_play_data")


def resolve_data_file(*relative_parts: str) -> Path:
    """Resolve data from the repo checkout first, then from installed package data."""
    local_candidate = PROJECT_ROOT.joinpath(*relative_parts)
    if local_candidate.exists():
        return local_candidate

    package_candidate = _package_root().joinpath(*relative_parts)
    return Path(str(package_candidate))


def resolve_asset_file(*relative_parts: str) -> Path:
    return resolve_data_file("assets", *relative_parts)


def resolve_reference_file(*relative_parts: str) -> Path:
    return resolve_data_file("reference_data", *relative_parts)


def resolve_game_catalog_file() -> Path:
    return resolve_reference_file("game_catalog.json")


def resolve_track_metadata_file() -> Path:
    return resolve_game_catalog_file()
