"""Tests for configuration system."""

import poolparty as pp


def test_config_defaults():
    """Test that Config has correct default values."""
    from poolparty.config import Config

    config = Config()

    assert config.suppress_styles is False
    assert config.suppress_cards is False
    assert config.progress_mode == "auto"


def test_party_default_config():
    """Test that Party initializes with default Config."""
    pp.init()
    party = pp.get_active_party()

    assert party._config is not None
    assert party._config.suppress_cards is False


def test_suppress_cards_property():
    """Test that Party.suppress_cards uses config."""
    pp.init()
    party = pp.get_active_party()

    assert party.suppress_cards is False

    party._config.suppress_cards = True
    assert party.suppress_cards is True
