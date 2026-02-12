# Configuration file for the Sphinx documentation builder.

import os
import subprocess
import datetime

# -- Project information

project = "PyOMP"
start_year = 2024
this_year = datetime.datetime.today().year
copyright_years = (
    str(start_year) if start_year == this_year else f"{start_year}-{this_year}"
)
copyright = f"{copyright_years}, PyOMP developers"
author = "Giorgis Georgakoudis"

try:
    # Prefer RTD provided version when available.
    release = os.environ.get("READTHEDOCS_VERSION")
    if not release:
        # Ensure tags are available in shallow or sparse CI clones
        subprocess.run(["git", "fetch", "--tags"], check=False)
        release = (
            subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"])
            .strip()
            .decode()
        )
except subprocess.CalledProcessError:
    release = os.environ.get("READTHEDOCS_VERSION", "latest")
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
