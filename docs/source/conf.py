# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# let sphinx import gnssvod
import os
import sys
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'gnssvod'
copyright = '2026, Vincent Humphrey'
author = 'Vincent Humphrey'
release = '2026.2.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb"
]

templates_path = ['_templates']
exclude_patterns = []
nb_execution_mode = "off"   # don't execute notebooks on build
nb_render_markdown = "myst"
myst_enable_extensions = ["colon_fence"]
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

