"""Tests for configuration system."""

import os
import tempfile

import poolparty as pp


def test_config_defaults():
    """Test that Config has correct default values."""
    from poolparty.config import Config

    config = Config()

    assert config.suppress_styles is False
    assert config.suppress_cards is False
    assert config.text_progress is True


def test_config_from_toml():
    """Test loading config from TOML file."""
    from poolparty.config import Config

    toml_content = """
[general]
suppress_styles = true
suppress_cards = true
text_progress = false
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content)
        f.flush()
        temp_path = f.name

    try:
        config = Config.from_toml(temp_path)

        # Check general settings
        assert config.suppress_styles is True
        assert config.suppress_cards is True
        assert config.text_progress is False
    finally:
        os.unlink(temp_path)


def test_party_default_config():
    """Test that Party initializes with default Config."""
    pp.init()
    party = pp.get_active_party()

    assert party._config is not None
    assert party._config.suppress_cards is False


def test_load_config_into_party():
    """Test loading config into active party."""
    toml_content = """
[general]
suppress_cards = true
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content)
        f.flush()
        temp_path = f.name

    try:
        pp.init()
        pp.load_config(temp_path)

        party = pp.get_active_party()
        assert party._config.suppress_cards is True
    finally:
        os.unlink(temp_path)


def test_suppress_cards_property():
    """Test that Party.suppress_cards uses config."""
    pp.init()
    party = pp.get_active_party()

    assert party.suppress_cards is False

    party._config.suppress_cards = True
    assert party.suppress_cards is True
