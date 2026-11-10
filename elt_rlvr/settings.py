"""Settings read from the process environment."""

import os

DATABASE_VARIABLE = "ELT_RLVR_DATABASE"


def destination_database() -> str:
    """The warehouse database that holds every namespace of this environment."""
    return os.environ.get(DATABASE_VARIABLE, "ELT_RLVR")
