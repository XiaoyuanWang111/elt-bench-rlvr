"""Run one group of rollouts on one task and print the rewards.

    # no credentials: the scripted reference policy on a local task
    python -m elt_rlvr.rollout task=storefront policy=reference

    # a Tinker model on a local task (needs TINKER_API_KEY)
    python -m elt_rlvr.rollout task=library policy=tinker model_name=Qwen/Qwen3.6-35B-A3B group_size=2

The transcript of every rollout (assistant text, tool calls, tool results, per-turn metrics) and the final rewards
are written as JSON Lines to ``log_path``.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Literal

import chz
import tinker
from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.completers import TinkerTokenCompleter, TokenCompleter
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.rl.types import Trajectory

from elt_rlvr.env import ELTEnvGroupBuilder, EpisodeConfig
from elt_rlvr.grading.grader import RewardConfig
from elt_rlvr.policies import ScriptedPolicy, reference_messages
from elt_rlvr.tasks import load_tasks, select_backend

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Command-line configuration of one group rollout."""

    task: str = "library"
    policy: Literal["reference", "tinker"] = "reference"
    model_name: str = "Qwen/Qwen3.6-35B-A3B"
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None
    group_size: int = 1
    max_tokens: int = 2048
    temperature: float = 1.0
    max_turns: int = 30
    max_trajectory_tokens: int = 48_000

    elt_bench_root: str | None = None
    ground_truth_root: str | None = None
    warehouse: str = "duckdb"
    cache_syncs: bool = True
    heldout_samples: int = 2

    log_path: str = "runs/rollout.jsonl"
    base_url: str | None = None


def _turn_records(index: int, trajectory: Trajectory) -> list[dict]:
    return [
        {"rollout": index, "turn": turn, "logs": transition.logs, "metrics": transition.metrics}
        for turn, transition in enumerate(trajectory.transitions)
    ]


async def run(cli_config: CLIConfig) -> list[dict]:
    """Run the rollouts and return one summary per rollout."""
    task = load_tasks(
        [cli_config.task],
        elt_bench_root=cli_config.elt_bench_root,
        ground_truth_root=cli_config.ground_truth_root,
        destination=cli_config.warehouse,
    )[0]
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(cli_config.model_name)
    builder = ELTEnvGroupBuilder(
        task,
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        group_size=cli_config.group_size,
        backend=select_backend(
            warehouse=cli_config.warehouse,
            elt_bench_root=cli_config.elt_bench_root,
            cache_syncs=cli_config.cache_syncs,
        ),
        reward=RewardConfig(heldout_samples=cli_config.heldout_samples),
        episode=EpisodeConfig(
            max_turns=cli_config.max_turns,
            max_trajectory_tokens=cli_config.max_trajectory_tokens,
            max_generation_tokens=cli_config.max_tokens,
        ),
    )

    envs = await builder.make_envs()
    policies: list[TokenCompleter]
    if cli_config.policy == "reference":
        renderer = get_renderer(renderer_name, tokenizer_utils.get_tokenizer(cli_config.model_name))
        policies = [ScriptedPolicy(renderer, reference_messages(task)) for _ in envs]
    else:
        service_client = tinker.ServiceClient(base_url=cli_config.base_url)
        sampling_client = await service_client.create_sampling_client(
            base_model=cli_config.model_name, model_path=cli_config.load_checkpoint_path
        )
        completer = TinkerTokenCompleter(
            sampling_client, max_tokens=cli_config.max_tokens, temperature=cli_config.temperature
        )
        policies = [completer for _ in envs]

    try:
        trajectories = list(
            await asyncio.gather(*[do_single_rollout(policy, env) for policy, env in zip(policies, envs, strict=True)])
        )
        graded = await builder.compute_group_rewards(trajectories, envs)
    finally:
        await builder.cleanup()

    summaries = [
        {
            "task": task.task_id,
            "rollout": index,
            "policy": cli_config.policy,
            "model": cli_config.model_name if cli_config.policy == "tinker" else None,
            "stop_reason": trajectory.stop_reason,
            "reward": reward,
            **metrics,
            # Every turn sends the whole conversation again, so prompt tokens add up over the turns.
            "prompt_tokens": sum(transition.ob.length for transition in trajectory.transitions),
            "sampled_tokens": sum(len(transition.ac.tokens) for transition in trajectory.transitions),
        }
        for index, (trajectory, (reward, metrics)) in enumerate(zip(trajectories, graded, strict=True))
    ]
    log_path = Path(cli_config.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        for index, trajectory in enumerate(trajectories):
            for record in _turn_records(index, trajectory):
                handle.write(json.dumps(record, default=str) + "\n")
        for summary in summaries:
            handle.write(json.dumps({"summary": summary}, default=str) + "\n")
    return summaries


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cli_config = chz.entrypoint(CLIConfig)
    for summary in asyncio.run(run(cli_config)):
        print(json.dumps(summary, default=str))
    print(f"transcript: {cli_config.log_path}")


if __name__ == "__main__":
    main()
