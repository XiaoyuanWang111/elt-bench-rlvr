"""The credential-free integration test: the full tinker-cookbook rollout path on DuckDB with scripted policies.

A scripted policy emits assistant turns as tokens. Everything after that is the production path: the renderer parses
the tool calls, the tool-use environment runs the tools, and the group builder grades the warehouse state.
"""

import json
import pickle

import pytest

pytest.importorskip("tinker_cookbook")

from support import open_session, reference_extract_load, reference_models
from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.data_processing import compute_advantages
from tinker_cookbook.rl.rollouts import do_group_rollout, do_single_rollout
from tinker_cookbook.rl.types import StopReason, TrajectoryGroup
from tinker_cookbook.tool_use.types import ToolInput

from elt_rlvr import rollout, train
from elt_rlvr.env import ELTDatasetBuilder, ELTEnvGroupBuilder, EpisodeConfig
from elt_rlvr.policies import (
    ScriptedPolicy,
    pipeline_messages,
    reference_messages,
    tool_call_message,
)
from elt_rlvr.tasks.local import LOCAL_TASKS
from elt_rlvr.tools import ELTTools

MODEL_NAME = "Qwen/Qwen3-8B"

pytestmark = pytest.mark.tokenizer


@pytest.fixture(scope="module")
def renderer():
    tokenizer = tokenizer_utils.get_tokenizer(MODEL_NAME)
    return get_renderer(model_info.get_recommended_renderer_name(MODEL_NAME), tokenizer)


def _builder(task, group_size=1, **episode):
    return ELTEnvGroupBuilder(
        task, model_name=MODEL_NAME, renderer_name=None, group_size=group_size, episode=EpisodeConfig(**episode)
    )


@pytest.mark.parametrize("task_id", list(LOCAL_TASKS))
async def test_reference_rollout_earns_full_reward(task_id, local_tasks, renderer):
    task = local_tasks[task_id]

    group = await do_group_rollout(_builder(task), ScriptedPolicy(renderer, reference_messages(task)))

    assert group.get_total_rewards() == [1.0]
    assert group.metrics_G[0]["solved"] == 1.0
    assert group.metrics_G[0]["submitted"] == 1.0
    assert group.trajectories_G[0].stop_reason == StopReason.TOOL_STOPPED


async def test_rollout_that_only_submits_earns_nothing(local_tasks, renderer):
    policy = ScriptedPolicy(renderer, [tool_call_message("submit", {})])

    group = await do_group_rollout(_builder(local_tasks["library"]), policy)

    assert group.get_total_rewards() == [0.0]
    assert len(group.trajectories_G[0].transitions) == 1


async def test_group_with_different_pipelines_gets_different_rewards_and_centered_advantages(local_tasks, renderer):
    task = local_tasks["library"]
    models = reference_models(task)
    wrong_column = {"member_activity": models["member_activity"].replace("l.days_late > 14", "l.days_late > 0")}
    scripts = [
        reference_messages(task),
        pipeline_messages(reference_extract_load(task), wrong_column),
        pipeline_messages(reference_extract_load(task), {}),
        [tool_call_message("submit", {})],
    ]
    builder = _builder(task, group_size=len(scripts))

    envs = await builder.make_envs()
    try:
        trajectories = [
            await do_single_rollout(ScriptedPolicy(renderer, script), env)
            for script, env in zip(scripts, envs, strict=True)
        ]
        graded = await builder.compute_group_rewards(trajectories, envs)
    finally:
        await builder.cleanup()

    rewards = [reward for reward, _ in graded]
    assert rewards == pytest.approx([1.0, 0.3 + 0.7 * 0.8, 0.3, 0.0])
    group = TrajectoryGroup(trajectories, rewards, [metrics for _, metrics in graded])
    advantages = compute_advantages([group])[0]
    assert advantages.sum().item() == pytest.approx(0.0, abs=1e-6)
    assert advantages[0] > advantages[1] > advantages[2] > advantages[3]


async def test_episode_stops_at_the_turn_limit_and_is_still_graded(local_tasks, renderer):
    task = local_tasks["library"]
    script = reference_messages(task)[:-1] + [tool_call_message("list_files", {})] * 10

    group = await do_group_rollout(_builder(task, max_turns=6), ScriptedPolicy(renderer, script))

    trajectory = group.trajectories_G[0]
    assert trajectory.stop_reason == StopReason.MAX_TURNS
    assert len(trajectory.transitions) == 6
    assert group.metrics_G[0]["submitted"] == 0.0
    assert group.get_total_rewards() == [1.0]


async def test_tool_errors_are_returned_to_the_policy_as_feedback(local_tasks, renderer):
    task = local_tasks["library"]
    script = [
        tool_call_message("read_file", {"path": "../grader/task.yaml"}),
        tool_call_message("query", {"sql": "select * from members"}),
        tool_call_message("write_file", {"path": "data_model.yaml", "content": "models: []"}),
        tool_call_message("submit", {}),
    ]

    group = await do_group_rollout(_builder(task), ScriptedPolicy(renderer, script))

    assert group.metrics_G[0]["tool_errors"] == 3
    logs = [transition.logs["tool_result_0"] for transition in group.trajectories_G[0].transitions]
    assert "outside the workspace" in logs[0]
    assert "unknown table" in logs[1]
    assert "you can write only" in logs[2]


async def test_cleanup_drops_namespaces_and_workspaces(local_tasks, renderer):
    task = local_tasks["library"]
    builder = _builder(task, group_size=2)
    envs = await builder.make_envs()
    sessions = list(builder._sessions.values())
    await do_single_rollout(ScriptedPolicy(renderer, reference_messages(task)), envs[0])

    await builder.cleanup()

    for session in sessions:
        assert session.warehouse.list_tables(session.namespace) == []
        assert not session.workspace.exists()


async def test_tool_specs_name_the_seven_tools(local_tasks, warehouse, workspace):
    tools = ELTTools(open_session(local_tasks["library"], warehouse, workspace)).all()

    assert [tool.name for tool in tools] == [
        "list_files",
        "read_file",
        "write_file",
        "run_extract_load",
        "run_transform",
        "query",
        "submit",
    ]
    assert all(tool.to_spec()["description"] for tool in tools)
    listing = await tools[0].run(ToolInput(arguments={}))
    assert "config.yaml" in listing.messages[0]["content"]
    assert (await tools[-1].run(ToolInput(arguments={}))).should_stop


def test_builders_survive_pickling(local_tasks):
    builder = _builder(local_tasks["library"], group_size=2)

    restored = pickle.loads(pickle.dumps(builder))

    assert restored.task == builder.task
    assert restored.group_size == 2


async def test_dataset_builder_cycles_tasks_into_batches(local_tasks):
    builder = ELTDatasetBuilder(tasks=list(local_tasks.values()), model_name=MODEL_NAME, batch_size=4, group_size=2)

    train, evaluation = await builder()

    assert [group.task.task_id for group in train.get_batch(0)] == ["library", "storefront", "clinic", "library"]
    assert all(group.group_size == 1 for group in evaluation.get_batch(0))


async def test_dataset_length_covers_every_task(local_tasks):
    builder = ELTDatasetBuilder(tasks=list(local_tasks.values()), model_name=MODEL_NAME, batch_size=2, group_size=1)

    train, _ = await builder()

    assert len(train) == 2
    assert [group.task.task_id for group in train.get_batch(1)] == ["clinic", "library"]


async def test_task_that_fills_two_slots_of_a_batch_runs_as_two_independent_groups(local_tasks):
    builder = ELTDatasetBuilder(tasks=[local_tasks["library"]], model_name=MODEL_NAME, batch_size=2, group_size=1)
    train, _ = await builder()
    first, second = train.get_batch(0)
    await first.make_envs()
    await second.make_envs()
    (session,) = second._sessions.values()

    # tinker-cookbook runs the groups of a batch at the same time and cleans each one up when it ends.
    await first.cleanup()

    assert session.workspace.is_dir()
    assert session.warehouse.list_tables(session.namespace) == []
    await second.cleanup()
    assert not session.workspace.exists()


async def test_reference_rollout_command_writes_rewards_and_a_transcript(tmp_path):
    log_path = tmp_path / "rollout.jsonl"
    config = rollout.CLIConfig(task="storefront", policy="reference", model_name=MODEL_NAME, log_path=str(log_path))

    summaries = await rollout.run(config)

    assert [summary["reward"] for summary in summaries] == [1.0]
    assert all(summaries[0]["source_tables"].values())
    assert all(right for columns in summaries[0]["model_columns"].values() for right in columns.values())
    assert summaries[0]["prompt_tokens"] > summaries[0]["sampled_tokens"] > 0
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert records[-1]["summary"]["solved"] == 1.0
    assert any("run_extract_load" in str(record.get("logs", {}).get("tool_call_0")) for record in records)


def test_training_command_builds_a_cookbook_config_over_the_local_tasks(tmp_path):
    config = train.build_config(
        train.CLIConfig(tasks="library,clinic", model_name=MODEL_NAME, group_size=3, log_path=str(tmp_path / "run"))
    )

    assert [task.task_id for task in config.dataset_builder.tasks] == ["library", "clinic"]
    assert config.dataset_builder.group_size == 3
    assert config.dataset_builder.backend.warehouse == "duckdb"
