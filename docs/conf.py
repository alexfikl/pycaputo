# https://www.sphinx-doc.org/en/master/usage/configuration.html
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

import os
import sys
from importlib import metadata

from docutils import nodes

# {{{ project information

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


def add_dataclass_annotation(app, name, obj, options, bases):
    from dataclasses import is_dataclass

    if is_dataclass(obj):
        # NOTE: this needs to be a string because `dataclass` is a function, not
        # a class, so Sphinx gets confused when it tries to insert it into the docs
        bases.append(":func:`dataclasses.dataclass`")

    if object in bases:
        # NOTE: not very helpful to show inheritance from "object"
        bases.remove(object)


def linkcode_resolve(domain, info):
    url = None
    if domain != "py" or not info["module"]:
        return url

    modname = info["module"]
    objname = info["fullname"]

    mod = sys.modules.get(modname)
    if not mod:
        return url

    obj = mod
    for part in objname.split("."):
        try:
            obj = getattr(obj, part)
        except Exception:
            return url

    import inspect

    try:
        filepath = "{}.py".format(os.path.join(*obj.__module__.split(".")))
    except Exception:
        return url

    if filepath is None:
        return url

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        return url
    else:
        linestart, linestop = lineno, lineno + len(source) - 1

    return "https://github.com/alexfikl/pycaputo/blob/main/src/{}#L{}-L{}".format(
        filepath, linestart, linestop
    )


def setup(app) -> None:
    (tmp_url,) = (
        v for k, v in m.items() if k.startswith("Project-URL") and "Repository" in v
    )
    project_url = tmp_url.split(" ")[-1]

    app.add_role("ghpr", autolink(f"{project_url}/pull/{{number}}"))
    app.add_role("ghissue", autolink(f"{project_url}/issues/{{number}}"))
    app.connect("autodoc-process-bases", add_dataclass_annotation)


# }}}

# {{{ general configuration

# needed extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
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
    "navigation_with_keys": True,
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

html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]

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
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3", None),
    "rich": ("https://rich.readthedocs.io/en/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

# }}}
