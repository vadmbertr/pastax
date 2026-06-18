"""Sphinx project that builds the pastax API reference and emits it as a
MyST-MD AST (``*.myst.json``) consumable by mystmd's TOC ``pattern:`` entry.

Build with::

    sphinx-build -b myst docs/_sphinx docs/_sphinx/_build/myst

The resulting JSON files are ingested by ``myst build --html`` so the API
section renders inside the same site (and the same theme) as the rest of
the documentation.
"""

from __future__ import annotations

import json
import pathlib
import re

project = "pastax"
author = "Vadim Bertrand"

extensions = [
    "sphinx_ext_mystmd",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

# Google-style docstrings; keep numpy off so we don't double-parse.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

autodoc_member_order = "bysource"
autodoc_typehints = "description"
# Render parameter defaults as their source token (``kernel=l2_distance``)
# rather than the evaluated repr (``kernel=<function l2_distance>``).
autodoc_preserve_defaults = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

master_doc = "api"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# sphinx-ext-mystmd renders each documented class/function as a
# ``definitionList`` (the signature lives in a ``definitionTerm`` carrying the
# anchor), never as a ``heading``. The mystmd book-theme builds its right-hand
# "On this page" outline from headings only, so the API pages list nothing but
# the module title and the docstring prose sections. We post-process the emitted
# AST to inject a heading above each top-level object so it shows in the outline.

# Exactly ``pastax.<module>.<object>`` — i.e. module-level classes/functions.
# Methods/attributes (``pastax.solver.euler.ode_step``) and synthetic anchors
# (``id0``, ``id1``, …) do not match and are left untouched.
_TOP_LEVEL_ID = re.compile(r"^pastax\.[^.]+\.[^.]+$")


def _desc_name(node):
    """Return the object's display name (e.g. ``Euler``) from the first
    ``span`` with the ``sphinx-desc-name`` class inside a ``definitionTerm``."""
    if isinstance(node, dict):
        cls = node.get("class")
        if node.get("type") == "span" and (
            cls == "sphinx-desc-name"
            or (isinstance(cls, list) and "sphinx-desc-name" in cls)
        ):
            return "".join(
                c.get("value", "")
                for c in node.get("children", []) or []
                if isinstance(c, dict)
            )
        for child in node.get("children", []) or []:
            found = _desc_name(child)
            if found:
                return found
    return ""


def _inject_into(children):
    """Insert outline headings before top-level object ``definitionList`` nodes
    in ``children`` (in place), recursing into nested children."""
    i = 0
    while i < len(children):
        node = children[i]
        if isinstance(node, dict):
            _inject_into(node.get("children", []) or [])
            term = (node.get("children") or [None])[0] if node.get("type") == "definitionList" else None
            if (
                isinstance(term, dict)
                and term.get("type") == "definitionTerm"
                and _TOP_LEVEL_ID.match(term.get("identifier") or "")
            ):
                identifier = term["identifier"]
                prev = children[i - 1] if i > 0 else None
                already = (
                    isinstance(prev, dict)
                    and prev.get("type") == "heading"
                    and prev.get("identifier") == identifier
                )
                if not already:
                    heading = {
                        "type": "heading",
                        "depth": 2,
                        "identifier": identifier,
                        "label": term.get("label"),
                        "children": [
                            {
                                "type": "inlineCode",
                                "value": _desc_name(term) or identifier.rsplit(".", 1)[-1],
                            }
                        ],
                    }
                    children.insert(i, heading)
                    i += 1
                # Move the anchor onto the heading so links keep resolving and we
                # don't emit a duplicate identifier (which fails ``--strict``).
                term.pop("identifier", None)
                term.pop("label", None)
        i += 1


def _fix_math_nodes(node):
    """Lift LaTeX from a math node's child text into its ``value``.

    sphinx-ext-mystmd emits ``math`` / ``inlineMath`` nodes carrying their LaTeX
    as a child ``text`` node and no top-level ``value``. mystmd's math transform
    reads ``node.value`` and calls ``.match()`` on it, so an absent ``value``
    crashes the HTML build with "Cannot read properties of undefined (reading
    'match')". Per the MyST spec these are literal nodes: move the text into
    ``value`` and drop the children."""
    if isinstance(node, dict):
        if node.get("type") in ("math", "inlineMath") and "value" not in node:
            node["value"] = "".join(
                c.get("value", "")
                for c in node.get("children", []) or []
                if isinstance(c, dict)
            )
            node["children"] = []
        for child in node.get("children", []) or []:
            _fix_math_nodes(child)
    elif isinstance(node, list):
        for child in node:
            _fix_math_nodes(child)


def _inject_outline_headings(app, exception):  # noqa: D401 - sphinx hook
    if exception is not None:
        return
    for path in pathlib.Path(app.outdir).glob("**/*.myst.json"):
        data = json.loads(path.read_text())
        root = data.get("mdast", data)
        before = json.dumps(root)
        _fix_math_nodes(root)
        _inject_into(root.get("children", []) or [])
        if json.dumps(root) != before:
            path.write_text(json.dumps(data))


# sphinx-ext-mystmd (as of 2026-05) does not implement a visitor for
# docutils' ``abbreviation`` nodes, which Sphinx's Python domain emits to
# render the ``/`` (PEP 570) and ``*`` (PEP 3102) parameter separators in
# function signatures. Patch a passthrough so the AST build doesn't crash.
def setup(app):  # noqa: D401 - sphinx hook
    from sphinx_ext_mystmd.transform import MySTNodeVisitor

    if not hasattr(MySTNodeVisitor, "visit_abbreviation"):
        def visit_abbreviation(self, node):
            return self.enter_myst_node(
                {"type": "span", "class": ["abbreviation"], "children": []}, node
            )
        MySTNodeVisitor.visit_abbreviation = visit_abbreviation

    app.connect("build-finished", _inject_outline_headings)
