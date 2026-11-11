"""The ELT environment as a tinker-cookbook RL dataset.

One ``ELTEnvGroupBuilder`` is one task. It opens ``group_size`` rollouts, each with its own workspace and warehouse
namespace, lets tinker-cookbook's tool-use environment drive them, and grades the whole group after the rollouts
end. Grading reads only the state each rollout left behind, so it does not depend on how the episode stopped.
"""

import asyncio
import logging
import math
import shutil
import tempfile
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import chz
from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.rl.rollout_limits import ParseErrorPolicy, RolloutLimits
from tinker_cookbook.rl.rollout_presets import RolloutConfig
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, Metrics, RLDataset, RLDatasetBuilder, Trajectory
from tinker_cookbook.tool_use import build_agent_tool_env

from elt_rlvr.backend import BackendConfig, build_engine, build_warehouse
from elt_rlvr.extract_load.base import ExtractLoadEngine
from elt_rlvr.grading.grader import Grade, RewardConfig, grade_group
from elt_rlvr.grading.reference import TaskReference
from elt_rlvr.prompts import system_prompt, task_message
from elt_rlvr.session import RolloutSession, SessionLimits
from elt_rlvr.task import TaskSpec
from elt_rlvr.tools import ELTTools
from elt_rlvr.warehouse import Warehouse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpisodeConfig:
    """Bounds of one episode."""

    max_turns: int = 30
    max_tool_calls: int = 60
    max_trajectory_tokens: int = 60_000
    max_generation_tokens: int = 8_192
    """Room for one turn. A reasoning model spends most of it before the tool call."""
    rollout_timeout_seconds: float = 1_800.0
    parse_error_retries: int = 2
    session_limits: SessionLimits = field(default_factory=SessionLimits)

    def rollout_config(self) -> RolloutConfig:
        return RolloutConfig(
            limits=RolloutLimits(
                max_turns=self.max_turns,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_tool_calls=self.max_tool_calls,
                rollout_timeout_seconds=self.rollout_timeout_seconds,
            ),
            parse_errors=ParseErrorPolicy(max_consecutive=self.parse_error_retries),
            # The tools change shared state (files, tables), so they run in the order the policy called them.
            tool_execution="sequential",
        )


@dataclass
class TaskRuntime:
    """What the rollouts of one task share inside a process: the warehouse, the engine and the reference state."""

    warehouse: Warehouse
    engine: ExtractLoadEngine
    reference: TaskReference


_runtimes: dict[tuple[str, str], TaskRuntime] = {}
_warehouses: dict[str, Warehouse] = {}
_runtime_lock = threading.Lock()


def task_runtime(task: TaskSpec, backend: BackendConfig) -> TaskRuntime:
    """The shared runtime of ``task`` on ``backend``, created on first use."""
    backend_key = repr(backend)
    with _runtime_lock:
        if (task.task_id, backend_key) not in _runtimes:
            if backend_key not in _warehouses:
                _warehouses[backend_key] = build_warehouse(backend)
            warehouse = _warehouses[backend_key]
            _runtimes[(task.task_id, backend_key)] = TaskRuntime(
                warehouse=warehouse,
                engine=build_engine(task, cache_syncs=backend.cache_syncs),
                reference=TaskReference(task, warehouse, build_engine(task, cache_syncs=False)),
            )
        return _runtimes[(task.task_id, backend_key)]


async def _no_step_reward(history: list[Message]) -> tuple[float, dict[str, float]]:
    """The episode-end reward of the tool environment. The reward comes from ``compute_group_rewards``."""
    return 0.0, {}


class ELTEnvGroupBuilder(EnvGroupBuilder):
    """Builds the rollouts of one ELT task and grades them."""

    def __init__(
        self,
        task: TaskSpec,
        *,
        model_name: str,
        renderer_name: str | None,
        group_size: int,
        backend: BackendConfig | None = None,
        reward: RewardConfig | None = None,
        episode: EpisodeConfig | None = None,
    ) -> None:
        self.task = task
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.group_size = group_size
        self.backend = backend or BackendConfig()
        self.reward = reward or RewardConfig()
        self.episode = episode or EpisodeConfig()
        self.grades: list[Grade] = []
        """The grades of the group graded last, in the order of its rollouts: what each reward is made of."""
        self._sessions: dict[int, RolloutSession] = {}

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_sessions"] = {}
        state["grades"] = []
        return state

    def fresh(self) -> "ELTEnvGroupBuilder":
        """A builder for the same task and settings that keeps track of its own rollouts."""
        return ELTEnvGroupBuilder(
            self.task,
            model_name=self.model_name,
            renderer_name=self.renderer_name,
            group_size=self.group_size,
            backend=self.backend,
            reward=self.reward,
            episode=self.episode,
        )

    async def make_envs(self) -> Sequence[Env]:
        runtime = task_runtime(self.task, self.backend)
        # Built before any rollout runs: a deployment that cannot load the reference fails here, not after the
        # policy has spent its turns.
        await asyncio.to_thread(runtime.reference.build)
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(self.model_name)
        renderer = get_renderer(renderer_name, tokenizer)
        prompt = system_prompt(self.backend.warehouse, runtime.warehouse.dialect)

        envs = []
        for _ in range(self.group_size):
            workspace = Path(tempfile.mkdtemp(prefix="elt_rlvr_"))
            session = await asyncio.to_thread(
                RolloutSession,
                task=self.task,
                warehouse=runtime.warehouse,
                engine=runtime.engine,
                workspace=workspace,
                limits=self.episode.session_limits,
            )
            tools = ELTTools(session).all()
            prefix = renderer.create_conversation_prefix_with_tools(
                tools=[tool.to_spec() for tool in tools], system_prompt=prompt
            )
            env = build_agent_tool_env(
                renderer=renderer,
                tools=tools,
                initial_messages=[*prefix, {"role": "user", "content": task_message(self.task)}],
                reward_fn=_no_step_reward,
                rollout_config=self.episode.rollout_config(),
                max_trajectory_tokens=self.episode.max_trajectory_tokens,
                max_generation_tokens=self.episode.max_generation_tokens,
                context_overflow_reward=0.0,
            )
            self._sessions[id(env)] = session
            envs.append(env)
        return envs

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        runtime = task_runtime(self.task, self.backend)
        sessions = [self._sessions[id(env)] for env in env_group]
        start = time.perf_counter()
        grades = await asyncio.to_thread(grade_group, sessions, runtime.reference, self.reward)
        grading_seconds = (time.perf_counter() - start) / max(len(sessions), 1)
        self.grades = grades

        results: list[tuple[float, Metrics]] = []
        for trajectory, session, grade in zip(trajectory_group, sessions, grades, strict=True):
            turns = len(trajectory.transitions)
            metrics: Metrics = grade.metrics()
            metrics.update(
                submitted=float(session.submitted),
                turns=turns,
                tool_calls=len(session.timings),
                tool_errors=session.tool_errors,
                extract_load_runs=session.extract_load_runs,
                connection_syncs=session.connection_syncs,
                environment_errors=session.environment_errors,
                tool_seconds=sum(timing.seconds for timing in session.timings),
                grading_seconds=grading_seconds,
            )
            results.append((grade.reward - self.reward.turn_cost * turns, metrics))
        return results

    async def cleanup(self) -> None:
        sessions, self._sessions = list(self._sessions.values()), {}
        for session in sessions:
            try:
                await asyncio.to_thread(session.close)
            except Exception as error:
                logger.warning("closing rollout %s failed: %s", session.namespace, error)
                shutil.rmtree(session.workspace, ignore_errors=True)

    def logging_tags(self) -> list[str]:
        return ["elt", self.task.task_id]


class ELTDataset(RLDataset):
    """Batches of task builders."""

    def __init__(self, builders: list[ELTEnvGroupBuilder], batch_size: int) -> None:
        self.builders = builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """The builders of one batch, wrapping around the task list.

        Every slot gets a builder of its own. The groups of a batch run at the same time and a builder's
        ``cleanup`` closes every rollout it made, so a task that fills two slots must not share one builder.
        """
        start = index * self.batch_size
        positions = range(start, start + self.batch_size)
        return [self.builders[position % len(self.builders)].fresh() for position in positions]

    def __len__(self) -> int:
        return math.ceil(len(self.builders) / self.batch_size)


@chz.chz
class ELTDatasetBuilder(RLDatasetBuilder):
    """Builds the training and evaluation datasets over a list of ELT tasks."""

    tasks: list[TaskSpec]
    model_name: str
    batch_size: int
    group_size: int
    renderer_name: str | None = None
    backend: BackendConfig = BackendConfig()
    reward: RewardConfig = RewardConfig()
    episode: EpisodeConfig = EpisodeConfig()

    def _builders(self, group_size: int) -> list[ELTEnvGroupBuilder]:
        return [
            ELTEnvGroupBuilder(
                task,
                model_name=self.model_name,
                renderer_name=self.renderer_name,
                group_size=group_size,
                backend=self.backend,
                reward=self.reward,
                episode=self.episode,
            )
            for task in self.tasks
        ]

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        return (
            ELTDataset(self._builders(self.group_size), self.batch_size),
            ELTDataset(self._builders(group_size=1), self.batch_size),
        )
