"""Tests for configuration system."""
import pytest
import tempfile
import os
import poolparty as pp


def test_config_defaults():
    """Test that Config has correct default values."""
    from poolparty.config import Config
    config = Config()
    
    assert config.remove_tags is True  # Default True for backwards compatibility
    assert config.suppress_styles is False
    assert config.suppress_cards is False
    assert config.show_name is True
    assert config.show_seq is True
    assert config.show_pool_seqs is True
    assert config.show_pool_states is True
    assert config.show_op_states is True


def test_config_from_toml():
    """Test loading config from TOML file."""
    from poolparty.config import Config
    
    toml_content = """
[general]
remove_tags = true
suppress_styles = true
suppress_cards = false

[columns]
name = true
seq = true
pool_seqs = false
pool_states = true
op_states = false

[design_cards.mutagenize]
positions = true
wt_chars = true
mut_chars = false
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        f.flush()
        temp_path = f.name
    
    try:
        config = Config.from_toml(temp_path)
        
        # Check general settings
        assert config.remove_tags is True
        assert config.suppress_styles is True
        assert config.suppress_cards is False
        
        # Check column settings
        assert config.show_name is True
        assert config.show_seq is True
        assert config.show_pool_seqs is False
        assert config.show_pool_states is True
        assert config.show_op_states is False
        
        # Check design card keys
        enabled = config.get_enabled_keys('mutagenize', ['positions', 'wt_chars', 'mut_chars'])
        assert 'positions' in enabled
        assert 'wt_chars' in enabled
        assert 'mut_chars' not in enabled
    finally:
        os.unlink(temp_path)


def test_config_get_enabled_keys():
    """Test get_enabled_keys with various scenarios."""
    from poolparty.config import Config
    
    config = Config()
    
    # No config for operation type - should return all keys
    all_keys = ['positions', 'wt_chars', 'mut_chars']
    enabled = config.get_enabled_keys('mutagenize', all_keys)
    assert enabled == set(all_keys)
    
    # Set specific keys
    config._design_cards['mutagenize'] = {'positions', 'wt_chars'}
    enabled = config.get_enabled_keys('mutagenize', all_keys)
    assert enabled == {'positions', 'wt_chars'}


def test_party_default_config():
    """Test that Party initializes with default Config."""
    pp.init()
    party = pp.get_active_party()
    
    assert party._config is not None
    assert party._config.show_name is True
    assert party._config.suppress_cards is False


def test_load_config_into_party():
    """Test loading config into active party."""
    toml_content = """
[general]
suppress_cards = true

[columns]
pool_seqs = false
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        f.flush()
        temp_path = f.name
    
    try:
        pp.init()
        pp.load_config(temp_path)
        
        party = pp.get_active_party()
        assert party._config.suppress_cards is True
        assert party._config.show_pool_seqs is False
    finally:
        os.unlink(temp_path)


def test_suppress_cards_property():
    """Test that Party.suppress_cards uses config."""
    pp.init()
    party = pp.get_active_party()
    
    assert party.suppress_cards is False
    
    party._config.suppress_cards = True
    assert party.suppress_cards is True


def test_design_card_filtering():
    """Test that design cards are filtered based on config."""
    toml_content = """
[design_cards.from_seqs]
seq_name = true
seq_index = false
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        f.flush()
        temp_path = f.name
    
    try:
        pp.init()
        pp.load_config(temp_path)
        
        pool = pp.from_seqs(['ACGT', 'TGCA'], seq_names=['s1', 's2'])
        df = pool.generate_library(num_seqs=2, report_design_cards=True)
        
        # seq_name should be present
        assert 'op[0]:from_seqs.key.seq_name' in df.columns
        # seq_index should be filtered out
        assert 'op[0]:from_seqs.key.seq_index' not in df.columns
    finally:
        os.unlink(temp_path)


def test_column_visibility_filtering():
    """Test that column visibility is controlled by config."""
    toml_content = """
[columns]
name = false
pool_states = false
op_states = false
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        f.flush()
        temp_path = f.name
    
    try:
        pp.init()
        pp.load_config(temp_path)
        
        pool = pp.from_seqs(['ACGT', 'TGCA']).mutagenize(num_mutations=1)
        df = pool.generate_library(num_seqs=2, report_design_cards=True)
        
        # name column should not be present
        assert 'name' not in df.columns
        # State columns should not be present
        state_cols = [c for c in df.columns if '.state' in c]
        assert len(state_cols) == 0
    finally:
        os.unlink(temp_path)


def test_no_config_shows_all():
    """Test that without config, all columns are shown."""
    pp.init()
    # Don't load any config
    
    pool = pp.from_seqs(['ACGT', 'TGCA'], seq_names=['s1', 's2'])
    df = pool.generate_library(num_seqs=2, report_design_cards=True)
    
    # All design card keys should be present
    assert 'op[0]:from_seqs.key.seq_name' in df.columns
    assert 'op[0]:from_seqs.key.seq_index' in df.columns
