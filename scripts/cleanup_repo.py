from __future__ import annotations

import argparse
import os
import stat
import shutil
from pathlib import Path


DEFAULT_REMOVE_FILES = [
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.prod.yml",
    "docker-compose.jenkins.yml",
    "Jenkinsfile",
    "Jenkinsfile.simple",
    "start-jenkins.bat",
    "JENKINSFILE_GUIDE.md",
    "JENKINSFILE_QUICKSTART.md",
    "DEPLOYMENT_GUIDE.md",
    "deploy.sh",
    "deploy.bat",
    "Procfile",
    "Makefile",
]

DEFAULT_REMOVE_DIRS = [
    ".pytest_cache",
    ".pytest_runtime",
    ".pytest_tmp",
    "pytest_tmp",
    "logs",
    ".github/workflows",
    "{data",
    # Frontend build artifacts and nested git metadata.
    "frontend/node_modules",
    "frontend/.git",
]


def _remove_path(root: Path, rel: str, removed: list[str], dry_run: bool) -> None:
    path = root / rel
    if not path.exists():
        return

    def onerror(func, target, exc_info):  # noqa: ANN001 - signature dictated by shutil
        try:
            os.chmod(target, stat.S_IWRITE)
            func(target)
        except Exception:
            raise

    if path.is_dir():
        if dry_run:
            removed.append(f"DIR  {rel}")
            return
        try:
            shutil.rmtree(path, ignore_errors=False, onerror=onerror)
            removed.append(f"DIR  {rel}")
        except Exception as exc:
            removed.append(f"FAIL DIR  {rel} ({type(exc).__name__}: {exc})")
        return

    if dry_run:
        removed.append(f"FILE {rel}")
        return
    try:
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        pass
    try:
        path.unlink(missing_ok=True)
        removed.append(f"FILE {rel}")
    except Exception as exc:
        removed.append(f"FAIL FILE {rel} ({type(exc).__name__}: {exc})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Cleanup repo artifacts and unused infra files.")
    parser.add_argument("--root", default=".", help="Repository root (default: current directory)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without deleting")
    parser.add_argument(
        "--output",
        default="CLEANUP_REMOVED.txt",
        help="Write removed paths to this file (relative to root)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    removed: list[str] = []

    for rel in DEFAULT_REMOVE_FILES:
        _remove_path(root, rel, removed, args.dry_run)

    # Delete pytest cache files created as folders like `pytest-cache-files-xxxx`.
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("pytest-cache-files-"):
            _remove_path(root, child.name, removed, args.dry_run)

    for rel in DEFAULT_REMOVE_DIRS:
        _remove_path(root, rel, removed, args.dry_run)

    removed_sorted = sorted(removed)
    out_path = root / args.output
    out_path.write_text("\n".join(removed_sorted) + ("\n" if removed_sorted else ""), encoding="utf-8")

    print(f"{'DRY RUN: would remove' if args.dry_run else 'Removed'} {len(removed_sorted)} paths.")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
