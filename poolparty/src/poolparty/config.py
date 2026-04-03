"""Configuration system for poolparty."""

from .types import beartype

VALID_PROGRESS_MODES = ("text", "auto")


@beartype
class Config:
    """Unified configuration for poolparty library output and behavior."""

    def __init__(self):
        self.suppress_styles: bool = False
        self.suppress_cards: bool = False
        self.progress_mode: str = "auto"

    def __repr__(self) -> str:
        return (
            f"Config(suppress_styles={self.suppress_styles}, suppress_cards={self.suppress_cards})"
        )
