# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Slurmise"
copyright = "2024, Troy Comi, David Turner, Giannis Paraskevakos, Rob Bierman"
author = "Troy Comi, David Turner, Giannis Paraskevakos, Rob Bierman"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.imgconverter",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
templates_path = ["_templates"]

# Development banner — set DOCS_DEV_BANNER=1 to display a notice at the top of every page.
# Enabled automatically by `poe docs-dev` and the docs CI on non-main branches.
html_context = {"dev_banner": bool(os.environ.get("DOCS_DEV_BANNER"))}
