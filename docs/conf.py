# https://www.sphinx-doc.org/en/master/usage/configuration.html
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

# {{{ project information

from importlib import metadata

m = metadata.metadata("pycaputo")
project = m["Name"]
author = m["Author"]
copyright = f"2023 {author}"
version = m["Version"]
release = version

# }}}

# {{{ general configuration

# needed extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

# extension for source files
source_suffix = ".rst"
# name of the main (master) document
master_doc = "index"
# min sphinx version
needs_sphinx = "4.0"
# files to ignore
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
# highlighting
pygments_style = "sphinx"

html_theme = "sphinx_rtd_theme"

# }}}

# {{{ internationalization

language = "en"

# }}

# {{{ extension settings

autoclass_content = "class"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": None,
    "show-inheritance": None,
}

# }}}

# {{{ links

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "rich": ("https://rich.readthedocs.io/en/stable", None),
}

# }}}
