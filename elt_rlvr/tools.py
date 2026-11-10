"""The seven tools the policy acts with, as tinker-cookbook tools over one :class:`RolloutSession`."""

import asyncio
import json
from collections.abc import Callable
from typing import Annotated, Any

from tinker_cookbook.tool_use import Tool, ToolResult, simple_tool_result, tool

from elt_rlvr.session import RolloutSession

MAX_RESULT_CHARS = 16_000


class ELTTools:
    """Tool methods bound to one rollout.

    The session's work is blocking (file and warehouse calls), so each call runs in a worker thread. A lock keeps
    the calls of one rollout in the order the policy issued them.
    """

    def __init__(self, session: RolloutSession) -> None:
        self._session = session
        self._lock = asyncio.Lock()

    async def _run(
        self, action: Callable[..., dict[str, Any]], *arguments: Any, should_stop: bool = False
    ) -> ToolResult:
        async with self._lock:
            result = await asyncio.to_thread(action, *arguments)
        text = json.dumps(result, default=str)
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + " ...[truncated]"
        return simple_tool_result(text, should_stop=should_stop)

    @tool
    async def list_files(
        self,
        path: Annotated[str, "Directory to list, relative to the workspace root."] = ".",
    ) -> ToolResult:
        """List the files and directories in a workspace directory."""
        return await self._run(self._session.list_files, path)

    @tool
    async def read_file(
        self,
        path: Annotated[str, "File to read, relative to the workspace root."],
        start_line: Annotated[int, "First line to return, starting at 1."] = 1,
        max_lines: Annotated[int, "Number of lines to return (at most 200)."] = 200,
    ) -> ToolResult:
        """Read a workspace file: config.yaml, data_model.yaml, schemas/*.csv, documentation/*.md or a pipeline file."""
        return await self._run(self._session.read_file, path, start_line, max_lines)

    @tool
    async def write_file(
        self,
        path: Annotated[str, "pipeline/extract_load.yaml or pipeline/models/<model>.sql"],
        content: Annotated[str, "Full new content of the file. Empty content deletes the file."],
    ) -> ToolResult:
        """Create or replace a pipeline file.

        Only pipeline/extract_load.yaml and pipeline/models/<model>.sql are writable.
        """
        return await self._run(self._session.write_file, path, content)

    @tool
    async def run_extract_load(
        self,
        connections: Annotated[
            list[str] | None, "Connections to sync again in full. Omit to apply the whole file."
        ] = None,
    ) -> ToolResult:
        """Apply pipeline/extract_load.yaml: extract the streams of its connections and load them into the warehouse.

        A stream that has already loaded from its connection with the same source configuration and sync mode is
        left alone, so after fixing one connection you can call this again. Name connections in `connections` to
        sync them again in full; in an append mode that loads their rows a second time.
        Returns, per stream, the rows loaded or the error, and the row count of every source table loaded so far.
        """
        return await self._run(self._session.run_extract_load, connections)

    @tool
    async def run_transform(
        self,
        models: Annotated[list[str] | None, "Model names to run with their dependencies. Omit to run all."] = None,
    ) -> ToolResult:
        """Run the SQL files in pipeline/models/ in dependency order and build one table per file.

        Returns, per model, its row count and columns, or the error.
        """
        return await self._run(self._session.run_transform, models)

    @tool
    async def query(
        self,
        sql: Annotated[str, "One SELECT statement over your source tables and models, by bare table name."],
    ) -> ToolResult:
        """Run a read-only SELECT in the warehouse and return the first rows."""
        return await self._run(self._session.query, sql)

    @tool
    async def submit(self) -> ToolResult:
        """Submit the files under pipeline/ as the final pipeline and end the episode."""
        return await self._run(self._session.submit, should_stop=True)

    def all(self) -> list[Tool]:
        return [
            self.list_files,
            self.read_file,
            self.write_file,
            self.run_extract_load,
            self.run_transform,
            self.query,
            self.submit,
        ]
