from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()
pname = "ontocast"

for path in sorted(Path(pname).rglob("*.py")):
    module_path = path.relative_to(pname).with_suffix("")
    parts = list(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        if not parts:
            continue
        # Package page at reference/<pkg>.md — not reference/<pkg>/__init__.md —
        # so identifiers are not registered twice when a stub already exists.
        doc_path = Path(*parts).with_suffix(".md")
    else:
        doc_path = path.relative_to(pname).with_suffix(".md")

    full_doc_path = Path("reference", doc_path)
    parts_str: tuple[str, ...] = tuple(parts)
    nav[parts_str] = str(full_doc_path)

    with mkdocs_gen_files.open(full_doc_path, "w") as f:
        ident = ".".join([pname] + parts)
        f.write(f"# `{ident}`\n\n::: {ident}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)
