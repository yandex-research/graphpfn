# NOTE: this file must not import anything from lib.

import os
from pathlib import Path

_PROJECT_DIR: None | Path = None


def get_project_dir() -> Path:
    global _PROJECT_DIR

    if _PROJECT_DIR is None:
        path = Path.cwd().resolve()
        pyproject_path = path / "pyproject.toml"
        while str(path) != path.root and not pyproject_path.exists():
            path = path.parent
            pyproject_path = path / "pyproject.toml"

        if pyproject_path.exists():
            if pyproject_path.parent != Path(__file__).resolve().parent.parent:
                raise RuntimeError(
                    "Failed to find the project directory. "
                    f" Most likely, you are running the code"
                    " in a virtual environment of a different project,"
                    f" namely, of this one: {pyproject_path.parent}"
                )
            _PROJECT_DIR = path
        else:
            raise RuntimeError(
                "Failed to find the project directory."
                " Most likely, you are outside of the project directory."
            )
    return _PROJECT_DIR


def get_cache_dir() -> Path:
    path = get_project_dir() / "cache"
    path.mkdir(exist_ok=True)
    return path


def get_data_dir() -> Path:
    return get_project_dir() / "data"


def get_exp_dir() -> Path:
    return get_project_dir() / "exp"


def get_local_dir() -> Path:
    return get_project_dir() / "local"


def is_local() -> bool:
    return not bool(os.environ.get("SNAPSHOT_PATH"))
