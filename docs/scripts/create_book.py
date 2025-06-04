from pathlib import Path


"""Utility to compile educational documentation into ``BOOK.md``.

This script walks through the ``learn`` and ``tutorials`` directories under
``docs/docs`` and concatenates every ``.md`` and ``.ipynb`` file it finds. If
``nbconvert`` is available, notebooks are converted to Markdown; otherwise a
placeholder note is inserted.
"""

try:
    import nbformat
    from nbconvert import MarkdownExporter
except Exception:  # pragma: no cover - optional dependency
    nbformat = None
    MarkdownExporter = None


def convert_ipynb_to_md(path: Path) -> str:
    """Return notebook ``path`` converted to Markdown."""

    if nbformat is None or MarkdownExporter is None:
        return (
            f"*Notebook {path.name} not converted: nbconvert not installed.*\n"
        )

    nb = nbformat.read(path, as_version=4)
    exporter = MarkdownExporter()
    body, _ = exporter.from_notebook_node(nb)
    return body


def assemble_book(output_path: str = "docs/BOOK.md") -> None:
    """Generate a Markdown book at ``output_path``."""

    docs_root = Path(__file__).resolve().parent.parent / "docs"
    sections = [docs_root / "learn", docs_root / "tutorials"]
    with open(output_path, "w", encoding="utf-8") as out:
        out.write("# DSPy Educational Book\n\n")

        for section in sections:
            if not section.exists():
                continue

            out.write(f"\n## {section.name.capitalize()}\n")

            for path in sorted(section.rglob("*")):
                if path.suffix.lower() == ".md":
                    out.write(f"\n### {path.relative_to(docs_root)}\n\n")
                    out.write(path.read_text(encoding="utf-8"))
                    out.write("\n")
                elif path.suffix.lower() == ".ipynb":
                    out.write(f"\n### {path.relative_to(docs_root)}\n\n")
                    out.write(convert_ipynb_to_md(path))
    print(f"Book assembled at {output_path}")


if __name__ == "__main__":  # pragma: no cover - manual script
    assemble_book()
