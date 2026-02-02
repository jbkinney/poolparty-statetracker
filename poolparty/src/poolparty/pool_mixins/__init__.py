"""Pool mixin classes for operation methods."""

# Generic mixins (used by base Pool)
from .common_ops_mixin import CommonOpsMixin
from .generic_fixed_ops_mixin import GenericFixedOpsMixin
from .region_ops_mixin import RegionOpsMixin
from .scan_ops_mixin import ScanOpsMixin
from .state_ops_mixin import StateOpsMixin

# Sequence-type specific mixins
from .dna_mixin import DnaMixin
from .protein_mixin import ProteinMixin

# Legacy mixins (deprecated, kept for backward compatibility)
from .base_ops_mixin import BaseOpsMixin
from .fixed_ops_mixin import FixedOpsMixin
from .orf_ops_mixin import OrfOpsMixin

__all__ = [
    # Generic mixins
    "CommonOpsMixin",
    "GenericFixedOpsMixin",
    "ScanOpsMixin",
    "StateOpsMixin",
    "RegionOpsMixin",
    # Sequence-type specific mixins
    "DnaMixin",
    "ProteinMixin",
    # Legacy (deprecated)
    "BaseOpsMixin",
    "FixedOpsMixin",
    "OrfOpsMixin",
]
