"""Configuration system for poolparty."""
import sys
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib
from .types import Optional, Any, beartype


@beartype
class Config:
    """Unified configuration for poolparty library output and behavior."""
    
    def __init__(self):
        # General settings (migrated from Party._defaults)
        self.remove_tags: bool = True  # Default True for backwards compatibility
        self.suppress_styles: bool = False
        self.suppress_cards: bool = False
        
        # Column visibility (all default to True = show)
        self.show_name: bool = True
        self.show_seq: bool = True
        self.show_pool_seqs: bool = True
        self.show_pool_states: bool = True
        self.show_op_states: bool = True
        
        # Design card keys: dict maps factory_name -> set of enabled keys
        # If factory_name not in dict, all keys are enabled
        self._design_cards: dict[str, set[str]] = {}
    
    @classmethod
    def from_toml(cls, filepath: str) -> "Config":
        """Load config from TOML file.
        
        Args:
            filepath: Path to TOML configuration file.
        
        Returns:
            Config instance with loaded settings.
        """
        with open(filepath, 'rb') as f:
            data = tomllib.load(f)
        
        config = cls()
        
        # Load general settings
        if 'general' in data:
            general = data['general']
            config.remove_tags = general.get('remove_tags', False)
            config.suppress_styles = general.get('suppress_styles', False)
            config.suppress_cards = general.get('suppress_cards', False)
        
        # Load column visibility settings
        if 'columns' in data:
            columns = data['columns']
            config.show_name = columns.get('name', True)
            config.show_seq = columns.get('seq', True)
            config.show_pool_seqs = columns.get('pool_seqs', True)
            config.show_pool_states = columns.get('pool_states', True)
            config.show_op_states = columns.get('op_states', True)
        
        # Load design card settings
        if 'design_cards' in data:
            for factory_name, keys_dict in data['design_cards'].items():
                # Only include keys that are set to True
                enabled_keys = {key for key, enabled in keys_dict.items() if enabled}
                config._design_cards[factory_name] = enabled_keys
        
        return config
    
    def get_enabled_keys(self, factory_name: str, all_keys: list[str]) -> set[str]:
        """Get enabled design card keys for an operation type.
        
        Args:
            factory_name: The operation's factory_name (e.g., 'mutagenize').
            all_keys: All possible keys for this operation type.
        
        Returns:
            Set of enabled key names. Returns all_keys if no config for this type.
        """
        if factory_name not in self._design_cards:
            return set(all_keys)  # Default: show all
        return self._design_cards[factory_name]
    
    def is_key_enabled(self, factory_name: str, key: str, all_keys: list[str]) -> bool:
        """Check if a specific design card key is enabled.
        
        Args:
            factory_name: The operation's factory_name.
            key: The design card key to check.
            all_keys: All possible keys for this operation type.
        
        Returns:
            True if the key should be included in output.
        """
        return key in self.get_enabled_keys(factory_name, all_keys)
    
    def __repr__(self) -> str:
        return (
            f"Config(remove_tags={self.remove_tags}, "
            f"suppress_styles={self.suppress_styles}, "
            f"suppress_cards={self.suppress_cards})"
        )
