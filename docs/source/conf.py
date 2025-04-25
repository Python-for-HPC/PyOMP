# Configuration file for the Sphinx documentation builder.

import subprocess

# -- Project information

project = "PyOMP"
copyright = "2024, PyOMP developers"
author = "Giorgis Georgakoudis"

try:
    release = (
        subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"])
        .strip()
        .decode()
    )
except subprocess.CalledProcessError:
    release = "latest"
version = release

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"
