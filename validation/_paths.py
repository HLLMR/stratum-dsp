from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """
    Find the repository root by walking upward until Cargo.toml is found.

    This keeps scripts robust even if they move within the validation/ tree.
    """
    p = (start or Path(__file__)).resolve()
    for cand in [p] + list(p.parents):
        if (cand / "Cargo.toml").exists():
            return cand
    raise RuntimeError(f"Could not locate repo root from: {p}")


def resolve_data_path(data_path_arg: str, repo_root: Path) -> Path:
    """
    Resolve a --data-path argument.

    If the path is relative, it is resolved relative to repo_root.
    """
    p = Path(data_path_arg)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


