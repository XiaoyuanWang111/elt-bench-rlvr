"""Policies that need no model: a fixed script of assistant turns, and the script that solves a task.

They drive the environment through the same token interface as a sampled model, so they exercise rendering, tool-call
parsing, the tools and the grader. The integration tests and the ``--policy reference`` rollout use them.
"""

import json
from typing import Any

import tinker
from tinker_cookbook.completers import StopCondition, TokenCompleter, TokensWithLogprobs
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.renderers.base import Message, RenderContext, ToolCall

from elt_rlvr.pipeline import EXTRACT_LOAD_PATH, MODELS_DIRECTORY
from elt_rlvr.task import TaskSpec

FINAL_MESSAGE: Message = {"role": "assistant", "content": "The pipeline is complete."}


def tool_call_message(name: str, arguments: dict[str, Any], *, call_id: str = "call_0") -> Message:
    """An assistant turn that calls one tool."""
    call = ToolCall(function=ToolCall.FunctionBody(name=name, arguments=json.dumps(arguments)), id=call_id)
    return {"role": "assistant", "content": "", "tool_calls": [call]}


def pipeline_messages(extract_load: str | None, models: dict[str, str]) -> list[Message]:
    """The turns that write a pipeline, run it and submit it."""
    messages = []
    if extract_load is not None:
        messages.append(tool_call_message("write_file", {"path": EXTRACT_LOAD_PATH, "content": extract_load}))
        messages.append(tool_call_message("run_extract_load", {}))
    for name, sql in models.items():
        messages.append(tool_call_message("write_file", {"path": f"{MODELS_DIRECTORY}/{name}.sql", "content": sql}))
    if models:
        messages.append(tool_call_message("run_transform", {}))
    messages.append(tool_call_message("submit", {}))
    return messages


def reference_messages(task: TaskSpec) -> list[Message]:
    """The turns that submit the task's reference pipeline."""
    if task.reference_dir is None:
        raise ValueError(f"task {task.task_id} has no reference pipeline")
    extract_load = (task.reference_dir / "extract_load.yaml").read_text(encoding="utf-8")
    models = {
        path.stem: path.read_text(encoding="utf-8") for path in sorted((task.reference_dir / "models").glob("*.sql"))
    }
    return pipeline_messages(extract_load, models)


class ScriptedPolicy(TokenCompleter):
    """Emits a fixed list of assistant messages, one per turn, as tokens. Serves a single rollout."""

    def __init__(self, renderer: Renderer, messages: list[Message]) -> None:
        self._renderer = renderer
        self._messages = list(messages)
        self._turn = 0

    async def __call__(
        self, model_input: tinker.ModelInput, stop: StopCondition, *, max_tokens: int | None = None
    ) -> TokensWithLogprobs:
        message = self._messages[self._turn] if self._turn < len(self._messages) else FINAL_MESSAGE
        self._turn += 1
        rendered = self._renderer.render_message(message, RenderContext(idx=0, is_last=True))
        tokens = [token for chunk in rendered.output for token in chunk.tokens]
        return TokensWithLogprobs(tokens=tokens, maybe_logprobs=[0.0] * len(tokens))
