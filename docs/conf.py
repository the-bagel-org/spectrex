# Configuration file for the Sphinx documentation builder.
#
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Make the package importable from the src/ layout
sys.path.insert(0, os.path.abspath("../src"))
# Make local Sphinx extensions importable
sys.path.insert(0, os.path.abspath("_ext"))


# Register the astronomy author-year citation style plugin for sphinxcontrib-bibtex.
from astro_cite import (
    AstroAuthorYearReferenceStyle,  # noqa: E402    # defined in _ext/astro_cite.py
)
from sphinxcontrib.bibtex.plugin import register_plugin  # noqa: E402

register_plugin(
    "sphinxcontrib.bibtex.style.referencing",
    "astro",
    AstroAuthorYearReferenceStyle,
    force=True,
)

# Allow Sphinx to find documents outside its source directory (CHANGELOG, etc.)
# by symlinking or by using the rootdir as the project root.
# We use the sphinx option to extend the source root instead.

# -- Project information -----------------------------------------------------

project = "spectrex"
copyright = "2026, Seidel et al."
author = "Seidel et al."
release = "v0.1dev"

# -- General configuration ---------------------------------------------------

extensions = [
    # MyST: parse .md and .ipynb as documentation source
    "myst_nb",
    # Autodoc: pull docstrings from source
    "sphinx.ext.autodoc",
    # Autosummary: generate API docs from docstrings
    "sphinx.ext.autosummary",
    # Auto-label sections for cross-referencing
    "sphinx.ext.autosectionlabel",
    # Cross-references to numpy/scipy/astropy docs
    "sphinx.ext.intersphinx",
    # Render docstrings in NumPy style
    "sphinx.ext.napoleon",
    # Source links (viewcode)
    "sphinx.ext.viewcode",
    # Render matplotlib plots from code blocks
    "matplotlib.sphinxext.plot_directive",
    # Math support
    "sphinx.ext.mathjax",
    # UI extras
    "sphinx_copybutton",
    "sphinx_design",
    # BibTeX bibliography
    "sphinxcontrib.bibtex",
]

numpydoc_show_class_members = False
numfig = True  # enable numbered figures for {numref} role

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "ssp_basics_draft.md",
]

# -- MyST / myst-nb options --------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
    "attrs_inline",
]
myst_dmath_double_inline = True
myst_title_to_header = True  # promote frontmatter `title:` to a document H1
nb_execution_mode = "off"  # set to 'auto' to execute notebooks at build time

# -- Autodoc / autosummary options -------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autosummary_generate = True
autodoc_typehints = "description"
autosectionlabel_prefix_document = (
    True  # avoid duplicate label collisions across docs
)
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# -- Intersphinx mapping -----------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- Bibliography ------------------------------------------------------------

bibtex_bibfiles = ["main.bib"]
bibtex_default_style = (
    "unsrt"  # bibliography list formatting (pybtex: unsrt/plain/alpha)
)
bibtex_reference_style = (
    "astro"  # in-text citation label formatting (astronomy author-year)
)
bibtex_encoding = "utf-8"

# Suppress known noise from included 3rd-party files (CONTRIBUTING.md, etc.)
# suppress_warnings = [
#    "myst.header",           # H2-as-first-heading in included CONTRIBUTING.md
#    "myst.xref_missing",     # cross-doc eq/fig refs that don't resolve yet
#    "autosectionlabel.*",    # duplicate section labels across notebooks
# ]

# -- MathJax configuration ---------------------------------------------------
# Register custom LaTeX macros used across the notebooks and docstrings.
# These mirror the `math:` frontmatter keys in the .ipynb files (which are
# JupyTeX-only and are NOT picked up by Sphinx/MathJax automatically).

mathjax3_config = {
    "tex": {
        "macros": {
            "msun": r"\text{M}_{\odot}",
            "Lsun": r"\text{L}_{\odot}",
            "Zsun": r"\text{Z}_{\odot}",
            "dobs": r"\mathbf{d}_{\text{obs}}",
            "gvar": r"g_{\rm var}",
            "logvar": r"\log_{10} g_{\rm var}",
        }
    }
}

# -- Plot directive configuration --------------------------------------------
# Show the source code alongside each plot so readers can reproduce it.
plot_include_source = True
plot_html_show_source_link = True

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["assets"]
html_css_files = ["../assets/wide-screen.css"]
html_show_sourcelink = True
html_sourcelink_suffix = ""

html_theme_options = {
    "repository_url": "https://github.com/the-bagel-org/spectrex",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "show_navbar_depth": 2,
    "launch_buttons": {
        "notebook_interface": "classic",
    },
}

html_title = "spectrex"


# -- Workaround: make datetime.date JSON-serializable for myst-nb on Py 3.14 ---
# myst-nb parses YAML frontmatter with PyYAML which converts bare dates like
# `date: 2026-03-25` to datetime.date objects. nbformat then fails to
# serialize them. We patch the JSON encoder once at import time.
import datetime as _dt
import json as _json

_orig_default = _json.JSONEncoder.default


def _patched_default(self, obj):
    if isinstance(obj, _dt.date):
        return obj.isoformat()
    return _orig_default(self, obj)


_json.JSONEncoder.default = _patched_default  # pyright: ignore


def autodoc_skip_member(app, what, name, obj, skip, options):
    """
    Skip imported members from submodules to prevent documentation duplication.
    """
    if skip:
        return True

    # Only apply to module-level members
    if what != "module":
        return False

    # Get the documenter from the call stack
    import inspect

    frame = inspect.currentframe()
    try:
        # Walk up the call stack to find the ModuleDocumenter
        while frame:
            local_vars = frame.f_locals
            if "self" in local_vars:
                documenter = local_vars["self"]
                # Check if this is a ModuleDocumenter with the info we need
                if hasattr(documenter, "modname") and hasattr(
                    documenter, "object"
                ):
                    current_module = documenter.modname

                    # Check if obj comes from a submodule
                    if hasattr(obj, "__module__"):
                        obj_module = obj.__module__

                        # Skip if object is from a submodule
                        if (
                            obj_module != current_module
                            and obj_module.startswith(current_module + ".")
                        ):
                            return True
                    break
            frame = frame.f_back
    finally:
        del frame

    return False


# -- Inject cell tags from #| comments into cell.metadata.tags ---------------
# myst_nb reads cell.metadata.tags for directives like hide-input, but many
# notebooks (especially Jupytext-produced ones) store tags inline as:
#   #| tags: [hide-input]
# This hook parses those comments and writes them back into the .ipynb files
# so myst_nb can find them. It runs once at build start and is idempotent.

import glob as _glob
import json as _json_nb
import re as _re


def _inject_inline_tags(app):
    """Parse ``#| tags: [...]`` lines from code cell source into cell.metadata.tags."""
    import nbformat

    tag_pattern = _re.compile(r"^# ?\|\s*tags:\s*\[([^\]]*)\]", _re.MULTILINE)
    notebook_paths = _glob.glob(
        str(app.srcdir) + "/content/**/*.ipynb", recursive=True
    )

    for path in notebook_paths:
        nb = nbformat.read(path, as_version=4)
        changed = False
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            src = cell.source
            m = tag_pattern.search(src)
            if m:
                new_tags = [
                    t.strip().strip("\"'")
                    for t in m.group(1).split(",")
                    if t.strip()
                ]
                existing = list(cell.metadata.get("tags", []))
                merged = existing + [t for t in new_tags if t not in existing]
                if merged != existing:
                    cell.metadata["tags"] = merged
                    changed = True
        if changed:
            nbformat.write(nb, path)


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)
    app.connect("builder-inited", _inject_inline_tags)
