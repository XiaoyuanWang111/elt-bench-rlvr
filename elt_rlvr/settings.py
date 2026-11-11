"""Settings read from the process environment, and the directory this process writes derived files to."""

import atexit
import os
import shutil
import tempfile
import threading
from pathlib import Path

DATABASE_VARIABLE = "ELT_RLVR_DATABASE"


def destination_database() -> str:
    """The warehouse database that holds every namespace of this environment."""
    return os.environ.get(DATABASE_VARIABLE, "ELT_RLVR")


_process_directory: Path | None = None
_process_directory_lock = threading.Lock()


def process_directory(name: str) -> Path:
    """The directory ``name`` under a temporary directory that belongs to this process.

    Generated tasks and the workspaces derived from ELT-Bench tasks are written there by default. A directory per
    process keeps two runs on one machine, or a run and the tests, from rewriting each other's files. It is
    removed when the process exits.
    """
    global _process_directory
    with _process_directory_lock:
        if _process_directory is None:
            _process_directory = Path(tempfile.mkdtemp(prefix="elt_rlvr_"))
            atexit.register(shutil.rmtree, _process_directory, ignore_errors=True)
        return _process_directory / name
