"""Test configuration for statecounter test suite."""
from pathlib import Path
import sys

# Ensure the statecounter package root is on sys.path so imports like
# `import tests.ops...` (relative to this tests package) work when running
# from the workspace root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
