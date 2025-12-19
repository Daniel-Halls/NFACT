import os
import sys
from datetime import datetime


# path for images in Sphinx

# Add project root to path
sys.path.insert(0, os.path.abspath(".."))

project = "NFACT"
author = "Daniel Halls"
copyright = f"{datetime.now().year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = [
    "override.css",
]
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
}
