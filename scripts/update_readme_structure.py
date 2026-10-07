"""Regenerate the repository tree section in README.md."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
START = "<!-- BEGIN AUTO-GENERATED REPOSITORY STRUCTURE -->"
END = "<!-- END AUTO-GENERATED REPOSITORY STRUCTURE -->"
IGNORED_DIRECTORIES = {".git", "__pycache__", ".ipynb_checkpoints"}
IGNORED_FILES = {".DS_Store"}


def build_nested_tree(paths: list[Path]) -> dict:
    tree: dict = {}
    for path in paths:
        node = tree
        for part in path.parts[:-1]:
            node = node.setdefault(part, {})
        node[path.parts[-1]] = None
    return tree


def render_tree(node: dict, prefix: str = "") -> list[str]:
    entries = sorted(
        node.items(),
        key=lambda item: (item[1] is None, item[0].lower()),
    )
    lines: list[str] = []
    for index, (name, child) in enumerate(entries):
        last = index == len(entries) - 1
        connector = "└── " if last else "├── "
        if child is None:
            lines.append(f"{prefix}{connector}{name}")
        else:
            lines.append(f"{prefix}{connector}{name}/")
            next_prefix = prefix + ("    " if last else "│   ")
            lines.extend(render_tree(child, next_prefix))
    return lines


def repository_tree() -> str:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if path.is_file():
            if any(part in IGNORED_DIRECTORIES for part in relative.parts):
                continue
            if path.name in IGNORED_FILES:
                continue
            files.append(relative)

    tree = build_nested_tree(files)
    lines = [f"{ROOT.name}/"]
    lines.extend(render_tree(tree, ""))
    return "\n".join(lines)


def update_readme() -> None:
    content = README.read_text(encoding="utf-8")
    start = content.find(START)
    end = content.find(END)
    if start == -1 or end == -1 or end < start:
        raise SystemExit("README markers are missing or out of order")

    generated = (
        f"{START}\n"
        "```text\n"
        f"{repository_tree()}\n"
        "```\n"
        f"{END}"
    )
    updated = content[:start] + generated + content[end + len(END) :]
    README.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    update_readme()
