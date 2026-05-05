"""Astronomy author-year citation style for sphinxcontrib-bibtex.

Produces natbib-like in-text and parenthetical citations:

  - ``:cite:t:`key``` — textual:
      Smith (2020)  /  Smith & Jones (2020)  /
      Smith, Jones & Brown (2020)  /  Smith et al. (2020)

  - ``:cite:p:`key``` — parenthetical:
      (Smith, 2020)  /  (Smith & Jones, 2020)  /
      (Smith, Jones & Brown, 2020)  /  (Smith et al., 2020)

  - ``:cite:ts:`key``` / ``:cite:ps:`key``` — same with full author list.

Author count thresholds (configurable via ``AstroPersonStyle.max_names``):
  - ≤ max_names authors → list all names
  - > max_names authors → first author + *et al.*

Registration in conf.py::

    sys.path.insert(0, os.path.abspath("_ext"))

    from sphinxcontrib.bibtex.plugin import register_plugin
    from astro_cite import AstroAuthorYearReferenceStyle

    register_plugin(
        "sphinxcontrib.bibtex.style.referencing",
        "astro",
        AstroAuthorYearReferenceStyle,
        force=True,
    )
    bibtex_reference_style = "astro"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Union

from pybtex.richtext import Tag, Text
from pybtex.style.template import (
    FieldIsMissing,
    _format_list,  # noqa: PLC2701 – internal but stable across pybtex versions
    first_of,
    optional,
    tag,
)
from pybtex.style.template import (
    field as bib_field,
)
from sphinxcontrib.bibtex.style.referencing import BracketStyle, PersonStyle
from sphinxcontrib.bibtex.style.referencing.author_year import (
    AuthorYearReferenceStyle,
)
from sphinxcontrib.bibtex.style.template import join, node

if TYPE_CHECKING:
    from pybtex.richtext import BaseText


# ---------------------------------------------------------------------------
# Custom template nodes
# ---------------------------------------------------------------------------


@node
def names_astro(children, data, role, max_names=3, **kwargs):
    """Format persons with astronomy-style truncation.

    Lists every author when the total count is ≤ *max_names*.  When it
    exceeds *max_names*, only the first author is shown followed by the
    ``other`` string (usually *italic* "et al.").
    """
    assert not children

    try:
        persons = data["entry"].persons[role]
    except KeyError:
        raise FieldIsMissing(role, data["entry"])

    style = data["style"]
    # Each call returns a Node – not yet rendered BaseText.
    name_nodes = [
        style.person.style_plugin.format(person, style.person.abbreviate)
        for person in persons
    ]

    other = kwargs.pop("other", None)

    if other is not None and len(name_nodes) > max_names:
        # Render only the first author node, then append the et-al text.
        first = name_nodes[0].format_data(data)
        return Text(first, other)

    # ≤ max_names authors: let join() render all nodes with proper separators.
    return join(**kwargs)[name_nodes].format_data(data)


@node
def author_or_editor_or_title_astro(children, data, **kwargs):
    """Author/editor/title fallback that uses :func:`names_astro`."""
    assert not children
    return first_of[
        optional[names_astro("author", **kwargs)],
        optional[names_astro("editor", **kwargs)],
        tag("em")[bib_field("title")],
    ].format_data(data)


# ---------------------------------------------------------------------------
# Person style
# ---------------------------------------------------------------------------


@dataclass
class AstroPersonStyle(PersonStyle):
    """Astronomy person formatting: list up to *max_names*, then *et al.*

    Defaults match common AAS/A&A journal conventions:
      - Comma between three or more authors.
      - Ampersand between exactly two authors.
      - No Oxford comma or "and" before last author.
      - Italic *et al.* when author count exceeds *max_names*.
    """

    #: Maximum authors to list before switching to *first* et al.
    max_names: int = 3

    #: Separator between authors when listing three or more.
    sep: Union["BaseText", str] = ", "

    #: Separator between exactly two authors.
    sep2: Optional[Union["BaseText", str]] = " & "

    #: Separator before the last author when listing exactly three.
    last_sep: Optional[Union["BaseText", str]] = " & "

    #: Text appended when the author count exceeds *max_names*.
    other: Optional[Union["BaseText", str]] = field(
        default_factory=lambda: Text(" ", Tag("em", "et al."))
    )

    def names(self, role: str, full: bool):
        """Return a :func:`names_astro` template for the given person role."""
        return names_astro(
            role,
            max_names=self.max_names,
            sep=self.sep,
            sep2=self.sep2,
            last_sep=self.last_sep,
            other=None if full else self.other,
        )

    def author_or_editor_or_title(self, full: bool):
        """Return an :func:`author_or_editor_or_title_astro` template."""
        return author_or_editor_or_title_astro(
            max_names=self.max_names,
            sep=self.sep,
            sep2=self.sep2,
            last_sep=self.last_sep,
            other=None if full else self.other,
        )


# ---------------------------------------------------------------------------
# Reference style
# ---------------------------------------------------------------------------


@dataclass
class AstroAuthorYearReferenceStyle(AuthorYearReferenceStyle):
    """Astronomy (natbib-like) author-year citation style.

    Differences from the built-in ``author_year`` style:

    * Round brackets ``( )`` instead of square brackets.
    * Author lists up to three names before truncating to *et al.*
    * Ampersand between exactly two authors.

    Supported cite roles
    --------------------
    ``:cite:t:`key```
        Textual: Smith (2020) / Smith & Jones (2020) /
        Smith, Jones & Brown (2020) / Smith et al. (2020)
    ``:cite:p:`key```
        Parenthetical: (Smith, 2020) / (Smith & Jones, 2020) /
        (Smith, Jones & Brown, 2020) / (Smith et al., 2020)
    ``:cite:ts:`key``` / ``:cite:ps:`key```
        Same with full (untruncated) author lists.
    ``:cite:ct:`key``` / ``:cite:cp:`key```
        Capitalised first word of the author name.
    ``:cite:author:`key```
        Author names only (no year).
    ``:cite:year:`key```
        Year only (no author names).
    """

    #: Round brackets for parenthetical citations: (Author, year).
    bracket_parenthetical: BracketStyle = field(
        default_factory=lambda: BracketStyle(left="(", right=")")
    )

    #: Round brackets around the year in textual citations: Author (year).
    bracket_textual: BracketStyle = field(
        default_factory=lambda: BracketStyle(left="(", right=")")
    )

    #: Astronomy-style person formatting.
    person: AstroPersonStyle = field(default_factory=AstroPersonStyle)

    #: Separator between author list and year in parenthetical citations.
    author_year_sep: Union["BaseText", str] = ", "
