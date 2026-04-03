"""Configuration system for poolparty."""

import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib
from .types import beartype

VALID_PROGRESS_MODES = ("text", "auto")


@beartype
class Config:
    """Unified configuration for poolparty library output and behavior."""

    def __init__(self):
        # General settings
        self.suppress_styles: bool = False
        self.suppress_cards: bool = False
        self.progress_mode: str = "auto"

    @classmethod
    def from_toml(cls, filepath: str) -> "Config":
        """Load config from TOML file."""
        with open(filepath, "rb") as f:
            data = tomllib.load(f)

        config = cls()

        # Load general settings
        if "general" in data:
            general = data["general"]
            config.suppress_styles = general.get("suppress_styles", False)
            config.suppress_cards = general.get("suppress_cards", False)
            progress_mode = general.get("progress_mode", "auto")
            if progress_mode not in VALID_PROGRESS_MODES:
                raise ValueError(
                    f"progress_mode must be one of {VALID_PROGRESS_MODES}, got {progress_mode!r}"
                )
            config.progress_mode = progress_mode

        return config

    def __repr__(self) -> str:
        return (
            f"Config(suppress_styles={self.suppress_styles}, suppress_cards={self.suppress_cards})"
        )
