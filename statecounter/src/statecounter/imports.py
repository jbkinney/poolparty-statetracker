"""Centralized type definitions for poolparty."""
from typing import TypeAlias, Literal, Union, Optional
from collections.abc import Sequence, Callable
from beartype import beartype
from numbers import Real, Integral
from beartype.roar import BeartypeCallHintParamViolation

Counter_type: TypeAlias = "statecounter.counter.Counter"

import math
import numpy
import pandas

# Forward reference type aliases (resolve circular imports)
Counter_type: TypeAlias = "statecounter.counter.Counter"

__all__ = [
    'beartype',
    'Union',
    'Optional', 
    'Sequence',
    'Callable',
    'Literal',
    'Real',
    'Integral',
    'Counter_type',
    'math',
]
