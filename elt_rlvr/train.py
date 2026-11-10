"""Train a model on ELT tasks with Tinker.

Example (credential-free tasks, one optimization step):

    python -m elt_rlvr.train tasks=library,storefront group_size=4 groups_per_batch=2 max_steps=1
"""

import asyncio
from datetime import datetime

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl.train import Config, main

from elt_rlvr.env import ELTDatasetBuilder, EpisodeConfig
from elt_rlvr.grading.grader import RewardConfig, RewardShape
from elt_rlvr.tasks import load_tasks, select_backend


@chz.chz
class CLIConfig:
    """Command-line configuration of an ELT training run."""

    # Model
    model_name: str = "Qwen/Qwen3.6-35B-A3B"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None
    max_tokens: int = 8192
    temperature: float = 1.0

    # Tasks: local task names, or ELT-Bench task names when elt_bench_root is set.
    tasks: str = "library,storefront,clinic"
    elt_bench_root: str | None = None
    ground_truth_root: str | None = None
    warehouse: str = "duckdb"
    data_seed: int = 0
    data_scale: int = 1
    cache_syncs: bool = True

    # Episode
    max_turns: int = 30
    max_trajectory_tokens: int = 60_000

    # Reward
    reward_shape: RewardShape = "multiplicative"
    extract_load_weight: float = 0.3
    heldout_samples: int = 2
    turn_cost: float = 0.0

    # Training
    group_size: int = 4
    groups_per_batch: int = 4
    learning_rate: float = 1e-5
    num_epochs: int = 1
    max_steps: int | None = None
    remove_constant_reward_groups: bool = True

    # Logging and checkpoints
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    eval_every: int = 0
    save_every: int = 0
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def build_config(cli_config: CLIConfig) -> Config:
    """Translate the command line into tinker-cookbook's RL training configuration."""
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(cli_config.model_name)
    run_name = (
        f"elt-{cli_config.model_name.replace('/', '-')}-{cli_config.group_size}group-"
        f"{cli_config.groups_per_batch}batch-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )
    dataset_builder = ELTDatasetBuilder(
        tasks=load_tasks(
            [name for name in cli_config.tasks.split(",") if name],
            elt_bench_root=cli_config.elt_bench_root,
            ground_truth_root=cli_config.ground_truth_root,
            destination=cli_config.warehouse,
            seed=cli_config.data_seed,
            scale=cli_config.data_scale,
        ),
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        batch_size=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        backend=select_backend(
            warehouse=cli_config.warehouse,
            elt_bench_root=cli_config.elt_bench_root,
            cache_syncs=cli_config.cache_syncs,
        ),
        reward=RewardConfig(
            shape=cli_config.reward_shape,
            extract_load_weight=cli_config.extract_load_weight,
            heldout_samples=cli_config.heldout_samples,
            turn_cost=cli_config.turn_cost,
        ),
        episode=EpisodeConfig(
            max_turns=cli_config.max_turns,
            max_trajectory_tokens=cli_config.max_trajectory_tokens,
            max_generation_tokens=cli_config.max_tokens,
        ),
    )
    return Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        recipe_name="elt_bench_rlvr",
        renderer_name=renderer_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        num_epochs=cli_config.num_epochs,
        max_steps=cli_config.max_steps,
        remove_constant_reward_groups=cli_config.remove_constant_reward_groups,
        log_path=cli_config.log_path or cli_utils.runs_path("elt_rlvr", run_name),
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name or run_name,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        base_url=cli_config.base_url,
    )


async def cli_main(cli_config: CLIConfig) -> None:
    config = build_config(cli_config)
    cli_utils.check_log_dir(config.log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)
    await main(config)


if __name__ == "__main__":
    asyncio.run(cli_main(chz.entrypoint(CLIConfig)))
