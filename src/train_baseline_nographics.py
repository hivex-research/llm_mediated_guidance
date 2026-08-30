"""Retrain the NO_INTERVENTION baseline with no_graphics=True.

Why this exists
---------------
The archived runs are not a controlled comparison. Per each condition's stored
experiment_config.json, the baseline trained with `no_graphics: false` while all
four intervention conditions trained with `no_graphics: true`. Under
`-nographics` Unity renders a flat constant image, so the 42x42x3 camera carries
no information and the policy effectively sees only its 8-dim vector. The
baseline therefore trained with strictly more information than the arms it is
compared against.

This script retrains the baseline under the interventions' observation
conditions, holding everything else identical to the archived config:

    env_parameters : {"random_village_location": 0}   (agent_count left default)
    lr 0.005, lambda 0.95, gamma 0.99
    sgd_minibatch_size 900, train_batch_size 9000, num_sgd_iter 3, clip 0.2
    stop at 1e6 env steps  (== the 1,008,000 the archived runs reached)

Seeds run sequentially, and that is deliberate
----------------------------------------------
A single trial already uses 16 env runners (matching the archived runs), i.e.
16 concurrent Unity instances, which saturates a 16-core machine. Launching
several trials at once would just make each proportionally slower, so seeds run
one after another.

Each seed writes its own results/train/NO_INTERVENTION_NOGFX_seed<N>/ directory
and is complete on its own. That means you can stop this at any point and use
whatever seeds finished -- there is no all-or-nothing failure mode. At roughly
2.65 h per seed, budget ~13 h for 5 seeds and ~27 h for 10.

    python src/train_baseline_nographics.py --seeds 0 1 2 3 4

eval_no_intervention.py picks the results up with --all-roots.
"""

import argparse
import os
import sys
from functools import partial
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import ray
from ray.rllib.algorithms.ppo import PPOConfig

from train_pipeline import train_policy, default_path
from environments.AWS_MultiAgentEnv import AerialWildFireSuppressionEnv as Unity3DEnv
from sidechannels.metrics_sidechannel import CustomMetricsCallback

# Exactly the archived baseline config, with no_graphics flipped to True.
BASE_CONFIG = {
    "name": "NO_INTERVENTION_NOGFX",
    "env_parameters": {"random_village_location": 0},
    "no_graphics": True,  # <-- the only intended difference
    "intervention_type": "none",
    "lr": 0.005,
    "lambda_": 0.95,
    "gamma": 0.99,
    "sgd_minibatch_size": 900,
    "train_batch_size": 9000,
    "num_sgd_iter": 3,
    "clip_param": 0.2,
}


class Args:
    """Minimal stand-in for train_pipeline's argparse namespace."""

    def __init__(self, file_name, horizon, stop_timesteps, num_workers, env):
        self.file_name = file_name
        self.horizon = horizon
        self.stop_timesteps = stop_timesteps
        self.num_workers = num_workers
        self.env = env


def build_config(experiment_config, args):
    policies, policy_mapping_fn = Unity3DEnv.get_policy_configs_for_game(args.env)
    return (
        PPOConfig()
        .environment(
            "unity3d",
            env_config={
                "file_name": args.file_name,
                "episode_horizon": args.horizon,
            },
        )
        .framework("torch")
        .env_runners(
            num_env_runners=args.num_workers if args.file_name else 0,
            num_envs_per_env_runner=1,
            rollout_fragment_length=300,
            num_cpus_per_env_runner=1,
        )
        .training(
            lr=experiment_config["lr"],
            lambda_=experiment_config["lambda_"],
            gamma=experiment_config["gamma"],
            sgd_minibatch_size=experiment_config["sgd_minibatch_size"],
            train_batch_size=experiment_config["train_batch_size"],
            num_sgd_iter=experiment_config["num_sgd_iter"],
            clip_param=experiment_config["clip_param"],
            model={"fcnet_hiddens": [256, 256]},
        )
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        .resources(num_gpus=args.num_gpus, num_cpus_for_local_worker=1)
        .callbacks(
            partial(
                CustomMetricsCallback,
                train_batch_size=experiment_config["train_batch_size"],
            )
        )
        .reporting(
            metrics_num_episodes_for_smoothing=10,
            min_time_s_per_iteration=10,
        )
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(10)),
        help="Seed indices to train in this process (default 0..9).",
    )
    p.add_argument("--file-name", type=str, default=str(default_path))
    p.add_argument("--horizon", type=int, default=3000)
    p.add_argument("--stop-timesteps", type=int, default=1_000_000)
    # 16 matches train_pipeline.py's default, i.e. what the archived runs used.
    # Lowering it changes how fast experience is collected, not just wall-time.
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--env", type=str, default="AerialWildfireSuppression")
    p.add_argument(
        "--num-gpus",
        type=float,
        default=1.0,
        help="Lower this (e.g. 0.25) only if sharing one Ray instance.",
    )
    opts = p.parse_args()

    current_dir = os.path.abspath(os.getcwd())
    ray.init(
        local_mode=False,
        runtime_env={
            "env_vars": {"PYTHONPATH": current_dir},
            "working_dir": current_dir,
            "excludes": [
                ".git/",
                "*.pyc",
                "__pycache__/",
                "env/",
                ".venv/",
                "results/",
            ],
        },
    )

    try:
        for seed in opts.seeds:
            experiment_config = dict(BASE_CONFIG)
            experiment_config["env_parameters"] = dict(BASE_CONFIG["env_parameters"])
            experiment_config["name"] = f"{BASE_CONFIG['name']}_seed{seed}"

            args = Args(
                file_name=opts.file_name,
                horizon=opts.horizon,
                stop_timesteps=opts.stop_timesteps,
                num_workers=opts.num_workers,
                env=opts.env,
            )
            args.num_gpus = opts.num_gpus

            print(f"\n=== training {experiment_config['name']} ===")
            config = build_config(experiment_config, args)
            config["intervention_type"] = experiment_config["intervention_type"]
            train_policy(experiment_config, config, args)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
