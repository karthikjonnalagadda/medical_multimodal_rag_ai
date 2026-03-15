from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_EXCLUDES = {
    ".git",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".pytest_runtime",
    ".pytest_tmp",
    "pytest_tmp",
    "node_modules",
}


def walk_tree(root: Path, excludes: set[str]) -> tuple[list[str], list[str]]:
    dirs: list[str] = []
    files: list[str] = []

    for current_root, dirnames, filenames in os.walk(root, topdown=True):
        cur_path = Path(current_root)
        rel_cur = cur_path.relative_to(root).as_posix()

        # Prune excluded dirs early.
        pruned: list[str] = []
        for name in list(dirnames):
            if name in excludes or name.startswith("pytest-cache-files-"):
                pruned.append(name)
        for name in pruned:
            dirnames.remove(name)

        if rel_cur != ".":
            dirs.append(rel_cur + "/")

        for fname in sorted(filenames):
            if fname.endswith((".pyc", ".pyo")):
                continue
            files.append((cur_path / fname).relative_to(root).as_posix())

    return sorted(dirs), sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a clean project tree listing.")
    parser.add_argument("--root", default=".", help="Project root")
    parser.add_argument("--output", default="PROJECT_TREE_CLEAN.txt", help="Output file name")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    dirs, files = walk_tree(root, DEFAULT_EXCLUDES)
    out = root / args.output

    lines: list[str] = []
    lines.append(f"ROOT {root}")
    lines.append("")
    lines.append("DIRECTORIES")
    lines.extend(dirs)
    lines.append("")
    lines.append("FILES")
    lines.extend(files)
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)
    print(f"dirs={len(dirs)} files={len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

