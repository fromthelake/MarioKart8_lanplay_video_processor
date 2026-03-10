from importlib.resources import files
from pathlib import Path


def _package_root():
    return files("mk8_local_play_data")


def resolve_data_file(*relative_parts: str) -> Path:
    """Resolve data from the repo checkout first, then from installed package data."""
    project_root = Path(__file__).resolve().parent
    local_candidate = project_root.joinpath(*relative_parts)
    if local_candidate.exists():
        return local_candidate

    package_candidate = _package_root().joinpath(*relative_parts)
    return Path(str(package_candidate))


def resolve_asset_file(*relative_parts: str) -> Path:
    return resolve_data_file("assets", *relative_parts)


def resolve_track_metadata_file() -> Path:
    return resolve_data_file("track_metadata.json")
