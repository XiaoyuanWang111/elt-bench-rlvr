"""The extract-load interface: copy the streams of a configured source into a warehouse namespace."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from elt_rlvr.pipeline import ConnectionConfig
from elt_rlvr.warehouse.base import Warehouse

METADATA_COLUMN_PREFIX = "_airbyte_"
"""Columns the loader adds to every raw table; they carry no source data."""


class ConnectionRejectedError(ValueError):
    """The connection configuration is invalid or is not allowed for this task. Shown to the policy."""


@dataclass(frozen=True)
class StreamResult:
    """Outcome of syncing one stream."""

    connection: str
    stream: str
    table: str
    succeeded: bool
    rows_loaded: int = 0
    error: str | None = None


class ExtractLoadEngine(ABC):
    """Runs sync jobs. One engine instance serves many rollouts and must be thread-safe."""

    @abstractmethod
    def sync(self, connection: ConnectionConfig, *, warehouse: Warehouse, namespace: str) -> list[StreamResult]:
        """Run one sync of ``connection`` into ``namespace`` and report every stream.

        Raises:
            ConnectionRejectedError: The configuration is invalid; nothing was loaded.
        """

    def sync_variant(self, connection: ConnectionConfig) -> str:
        """Names whatever decides the result of syncing ``connection`` besides the connection itself.

        The sync cache keeps apart results that differ in it. The Airbyte engine returns the connector definition
        it is set to use for the source type.
        """
        return ""

    def close(self) -> None:  # noqa: B027
        """Release what the engine created outside the warehouse."""


def table_name_for_stream(stream: str) -> str:
    """Destination table of a stream: the stream name with every non-identifier character replaced by ``_``."""
    cleaned = "".join(character if character.isalnum() or character == "_" else "_" for character in stream)
    return cleaned if not cleaned[:1].isdigit() else f"_{cleaned}"
