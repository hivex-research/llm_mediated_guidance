# At the very top of train_pipeline.py, before ANY other imports
import sys
from pathlib import Path
from functools import partial

# Add the src directory to the path
src_dir = Path(__file__).parent
project_root = src_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

import argparse
import json
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from environments.AWS_MultiAgentEnv import AerialWildFireSuppressionEnv as Unity3DEnv
from ray.rllib.utils.metrics import NUM_ENV_STEPS_SAMPLED_LIFETIME
from pathlib import Path
import os

from ray.rllib.algorithms.ppo import PPOTorchPolicy
from policies.hivex_ppo_torch_policy import HIVEXPPOTorchPolicy
from ray.rllib.policy.policy import PolicySpec

default_path = (
    Path(__file__).parent.parent
    / "env/Hivex_AerialWildfireSuppression_win/Hivex_AerialWildfireSuppression.exe"
)

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="AerialWildfireSuppression")
parser.add_argument(
    "--file-name",
    type=str,
    default=str(default_path),
    help="The Unity3d binary (compiled) game, e.g. "
    "'/home/ubuntu/soccer_strikers_vs_goalie_linux.x86_64'. Use `None` for "
    "a currently running Unity3D editor.",
)
parser.add_argument(
    "--from-checkpoint",
    type=str,
    default=None,
    help="Full path to a checkpoint file for restoring a previously saved "
    "Algorithm state.",
)
parser.add_argument("--num-workers", type=int, default=16)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=9999, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=1000000, help="Number of timesteps to train."
)

parser.add_argument(
    "--stop-reward",
    type=float,
    default=9999.0,
    help="Reward at which we stop training.",
)
parser.add_argument(
    "--horizon",
    type=int,
    default=3000,
    help="The max. number of `step()`s for any episode (per agent) before "
    "it'll be reset again automatically.",
)
parser.add_argument(
    "--framework",
    type=str,
    default="torch",
    help="The DL framework specifier.",
)

from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel
from sidechannels.human_command_sidechannel import HumanCommandChannel
from sidechannels.metrics_sidechannel import CustomMetricsCallback

import yaml


def train_policy(experiment_config, config, args):

    # Set up channels
    env_params_channel = EnvironmentParametersChannel()
    for key, env_parameter in experiment_config["env_parameters"].items():
        env_params_channel.set_float_parameter(key, env_parameter)
    stats_channel = StatsSideChannel()
    human_intervention_channel = HumanCommandChannel()

    # Register the environment
    tune.register_env(
        "unity3d",
        lambda c: Unity3DEnv(
            run_config=experiment_config,
            file_name=c["file_name"],
            no_graphics=experiment_config["no_graphics"],
            episode_horizon=c["episode_horizon"],
            side_channels=[
                env_params_channel,
                stats_channel,
                human_intervention_channel,
            ],
        ),
    )

    # Set stopping criteria
    stop = {
        NUM_ENV_STEPS_SAMPLED_LIFETIME: args.stop_timesteps,
    }

    output_dir = (Path("results/train") / experiment_config["name"]).resolve()

    # Create the unique directory
    os.makedirs(output_dir, exist_ok=True)

    # Save the experiment_config to the output directory
    with open(os.path.join(output_dir, "experiment_config.json"), "w") as f:
        json.dump(experiment_config, f, indent=4)

    # Run the experiment
    tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            stop=stop,
            verbose=1,
            storage_path=output_dir,  # Save results in the output directory
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=5,
                checkpoint_at_end=True,
            ),
        ),
    ).fit()


if __name__ == "__main__":

    agent_counts = [3]  # , 4, 5, 6]

    experiment_config_dirs = [
        ### no intervention
        # "src/configs/training_config_no_intervention.yml",
        ### rule based controller
        # "src/configs/training_config_rule_based_llama_3.1_8b_instruct.yml",
        # "src/configs/training_config_rule_based_pharia_1_7b_control_aligned.yml",
        "src/configs/training_config_rule_based_deepseek_r1.yml",
        ### natural language controller
        # "src/configs/training_config_natural_language_pharia_1_7b_control_aligned.yml",
        # "src/configs/training_config_natural_language_llama_3.1_8b_instruct.yml",
        # "src/configs/training_config_natural_language_deepseek_r1.yml",
        ### human intervention
        # "src/configs/training_config_human_intervention.yml",
    ]

    current_dir = os.path.abspath(os.getcwd())
    ray.init(
        local_mode=False,
        runtime_env={
            "env_vars": {"PYTHONPATH": current_dir},
            "working_dir": current_dir,
            "excludes": [
                ".git/",  # Exclude git to reduce package size
                "*.pyc",
                "__pycache__/",
                "*.pyo",
                "*.pyd",
                ".Python",
                "env/",
                "venv/",
                "results/",  # Exclude results directory
            ],
            "pip": [],  # Add any specific packages if needed
        },
    )

    try:
        for agent_count in agent_counts:
            for experiment_config_dir in experiment_config_dirs:
                for i in range(7):

                    with open(experiment_config_dir, "r") as file:
                        experiment_config = yaml.safe_load(file)
                        experiment_config["env_parameters"]["agent_count"] = agent_count
                        experiment_config["name"] = (
                            f"{experiment_config['name']}_{agent_count}"
                        )

                    args = parser.parse_args()

                    print(
                        "num workers: " + str(args.num_workers if args.file_name else 0)
                    )

                    # Get policies (different agent types; "behaviors" in MLAgents) and
                    # the mappings from individual agents to Policies.
                    policies, policy_mapping_fn = (
                        Unity3DEnv.get_policy_configs_for_game(args.env)
                    )
                    config = (
                        PPOConfig()
                        .environment(
                            "unity3d",
                            env_config={
                                "file_name": args.file_name,
                                "episode_horizon": args.horizon,
                            },
                        )
                        .framework(
                            "torch"
                        )  # "tf" if args.env != "Pyramids" else "torch"
                        # For running in editor, force to use just one Worker (we only have
                        # one Unity running)!
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
                        .multi_agent(
                            policies=policies, policy_mapping_fn=policy_mapping_fn
                        )
                        .resources(
                            num_gpus=1.0,
                            num_cpus_for_local_worker=1,
                        )
                        # .env_runners(num_gpus_per_env_runner=1.0)
                        .callbacks(
                            partial(
                                CustomMetricsCallback,
                                train_batch_size=experiment_config["train_batch_size"],
                            )
                        )
                        .reporting(
                            metrics_num_episodes_for_smoothing=10,
                            min_time_s_per_iteration=10,  # Adjust based on your needs
                        )
                    )

                    config["intervention_type"] = experiment_config["intervention_type"]
                    train_policy(
                        experiment_config=experiment_config, config=config, args=args
                    )
    finally:
        ray.shutdown()
