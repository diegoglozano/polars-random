import tomllib
from pathlib import Path

from polars_random import __version__ as VERSION


def test_versions_match():
    cargo = Path().absolute() / "Cargo.toml"
    with open(cargo, "rb") as f:
        data = tomllib.load(f)
        cargo_version = data["package"]["version"]

    assert VERSION == cargo_version
