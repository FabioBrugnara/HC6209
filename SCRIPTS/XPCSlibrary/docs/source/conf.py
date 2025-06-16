# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'XPCS_library'
copyright = '2025, Fabio Brugnara'
author = 'Fabio Brugnara'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon"]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']


import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Adjust path as needed


# Use ReadTheDocs theme (similar to NumPy docs)
html_theme = "sphinx_rtd_theme"


# Napoleon settings to ensure proper NumPy-style formatting
napoleon_google_docstring = False  # Disable Google-style
napoleon_numpy_docstring = True  # Enable NumPy-style
napoleon_include_init_with_doc = True
napoleon_use_rtype = False
napoleon_use_param = False