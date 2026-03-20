"""Configuration system for poolparty."""

import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib
from .types import beartype


@beartype
class Config:
    """Unified configuration for poolparty library output and behavior."""

    def __init__(self):
        # General settings
        self.suppress_styles: bool = False
        self.suppress_cards: bool = False
        self.text_progress: bool = True

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
            config.text_progress = general.get("text_progress", False)

        return config

    def __repr__(self) -> str:
        return (
            f"Config(suppress_styles={self.suppress_styles}, suppress_cards={self.suppress_cards})"
        )
