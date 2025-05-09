# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import inspect
import subprocess
import sys
from datetime import date
from importlib import import_module

from intersphinx_registry import get_intersphinx_mapping
# from sphinx_gallery.sorting import FileNameSortKey

import ncrf

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NCRF'
author = 'Proloy Das and Christian Brodbeck'
copyright = f"{date.today().year}, {author}"  # noqa: A001
package = ncrf.__name__
gh_url = ""
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be extensions coming
# with Sphinx (named "sphinx.ext.*") or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    "numpydoc",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Sphinx will warn about all references where the target cannot be found.
nitpicky = True
nitpick_ignore = [("py:obj", "optional")]

# A list of ignored prefixes for module index sorting.
modindex_common_prefix = [f"{package}."]

# The name of a reST role (builtin or Sphinx extension) to use as the default role, that
# is, for text marked up `like this`. This can be set to 'py:obj' to make `filter` a
# cross-reference to the Python function “filter”.
default_role = "autolink"

# list of warning types to suppress
suppress_warnings = ["config.cache"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_title = project

# -- autosummary -----------------------------------------------------------------------
autosummary_generate = True

# -- autodoc ---------------------------------------------------------------------------
autodoc_typehints = "none"
autodoc_member_order = "groupwise"
autodoc_warningiserror = True
autoclass_content = "class"

# -- intersphinx -----------------------------------------------------------------------
intersphinx_mapping = get_intersphinx_mapping(
    packages={
        "matplotlib",
        "mne",
        "numpy",
        "pandas",
        "python",
        "scipy",
        "sklearn",
        "numba",
    }
)
intersphinx_mapping["eelbrain"] = ("https://eelbrain.readthedocs.io/en/stable", None)
intersphinx_timeout = 5

# x-ref
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    # Matplotlib
    "Axes": "matplotlib.axes.Axes",
    "Figure": "matplotlib.figure.Figure",
    # MNE
    "DigMontage": "mne.channels.DigMontage",
    "Epochs": "mne.Epochs",
    "Evoked": "mne.Evoked",
    "Info": "mne.Info",
    "Projection": "mne.Projection",
    "Raw": "mne.io.Raw",
    # Python
    "bool": ":class:`python:bool`",
    "Path": "pathlib.Path",
    "TextIO": "io.TextIOBase",
    # Eelbrain
    "NDVar": "eelbrain.NDVar",
    "case": "eelbrain.Case",
    "sensor": "eelbrain.Sensor",
    "time": "eelbrain.UTS"

}

# -- sphinxcontrib-bibtex --------------------------------------------------------------
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
suppress_warnings = ["bibtex.duplicate_citation", "config.cache"]