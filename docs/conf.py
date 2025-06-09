# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'JaxDecomp'
copyright = '2025, Wassim Kabalan, François Lanusse'
author = 'Wassim Kabalan, François Lanusse'
release = 'v0.2.7'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**/__pycache__', '**/*.pyc']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']


import os
import sys

sys.path.insert(0, os.path.abspath('../src'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'myst_parser',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

autosummary_generate = True

myst_enable_extensions = [
    'dollarmath',  # Enables $...$ and $$...$$
    'amsmath',  # Enables AMS math environments
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

html_theme = 'sphinx_rtd_theme'
