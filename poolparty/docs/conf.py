"""Sphinx configuration for PoolParty documentation."""

import os
import sys

# Add the package source to the path for autodoc
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

# -- Project information -----------------------------------------------------
project = "PoolParty"
copyright = "2025, Justin Kinney"
author = "Justin Kinney"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "operations.rst"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Create _static directory if it doesn't exist (avoids warning)
os.makedirs(os.path.join(os.path.dirname(__file__), "_static"), exist_ok=True)

# -- Extension configuration -------------------------------------------------

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstrings = True
napoleon_numpy_docstrings = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}

# Type hints settings
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "statetracker": ("https://statetracker.readthedocs.io/en/latest/", None),
}

# Suppress warnings for missing references to beartype types
nitpicky = False
