# https://www.sphinx-doc.org/en/master/usage/configuration.html
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

import os
import sys
from docutils import nodes

# NOTE: seems necessary for readthedocs.org because it doesn't `pip install -e`
sys.path.insert(0, os.path.abspath(".."))


# {{{ project information

from importlib import metadata

m = metadata.metadata("pycaputo")
project = m["Name"]
author = m["Author-email"]
copyright = f"2023 {author}"  # noqa: A001
version = m["Version"]
release = version

# }}}

# {{{ github roles


def autolink(pattern: str):
    def role(name, rawtext, text, lineno, inliner, options=None, context=None):
        if options is None:
            options = {}

        if context is None:
            context = []

        url = pattern.format(number=text)
        node = nodes.reference(rawtext, f"#{text}", refuri=url, **options)
        return [node], []

    return role


def setup(app) -> None:
    (tmp_url,) = (
        v for k, v in m.items() if k.startswith("Project-URL") and "Repository" in v
    )
    project_url = tmp_url.split(" ")[-1]

    app.add_role("ghpr", autolink(f"{project_url}/pull/{{number}}"))
    app.add_role("ghissue", autolink(f"{project_url}/issues/{{number}}"))


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

html_theme = "sphinx_book_theme"
html_title = project
html_theme_options = {
    "show_toc_level": 2,
    "use_source_button": True,
    "use_repository_button": True,
    "repository_url": "https://github.com/alexfikl/pycaputo",
    "repository_branch": "main",
    "icon_links": [
        {
            "name": "Release",
            "url": "https://github.com/alexfikl/pycaputo/releases",
            "icon": "https://img.shields.io/github/v/release/alexfikl/pycaputo",
            "type": "url",
        },
        {
            "name": "License",
            "url": "https://github.com/alexfikl/pycaputo/tree/main/LICENSES",
            "icon": "https://img.shields.io/badge/License-MIT-blue.svg",
            "type": "url",
        },
        {
            "name": "CI",
            "url": "https://github.com/alexfikl/pycaputo",
            "icon": "https://github.com/alexfikl/pycaputo/workflows/CI/badge.svg",
            "type": "url",
        },
        {
            "name": "Issues",
            "url": "https://github.com/alexfikl/pycaputo/issues",
            "icon": "https://img.shields.io/github/issues/alexfikl/pycaputo",
            "type": "url",
        },
    ],
}

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
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3", None),
    "rich": ("https://rich.readthedocs.io/en/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

# }}}
