import argparse
import os
import numpy as np
import json

# rllib
from ray.rllib.algorithms.ppo import PPO
import torch
import ray
from ray import air, tune
from ray.air.constants import TRAINING_ITERATION
from ray.rllib.algorithms.ppo import PPOConfig
from environments.AWS_MultiAgentEnv import AerialWildFireSuppressionEnv as Unity3DEnv
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from ray.rllib.utils.test_utils import check_learning_achieved

# import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="AerialWildfireSuppression")
parser.add_argument(
    "--file-name",
    type=str,
    default="C:/Users/pdsie/Documents/human_intervention_marl/env/Hivex_AerialWildfireSuppression_win/Hivex_AerialWildfireSuppression.exe",
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
parser.add_argument("--num-workers", type=int, default=0)
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
# 1800000
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


def calculate_average(data_list):
    # Extract the first elements (the numeric values) from the tuples
    values = [item[0] for item in data_list]

    # Calculate the average if there are values in the list
    if len(values) > 0:
        return sum(values) / len(values)
    else:
        return 0.0  # Return 0 if the list is empty


def flatten_obs(obs):
    flattened_obs = {}
    for agent_id, agent_obs in obs.items():
        # agent_obs[0] is the visual observation (3D array), agent_obs[1] is the vector observation (1D array)
        vis_obs = agent_obs[0].flatten()  # Flatten the visual observation
        vec_obs = agent_obs[1]

        # Concatenate the flattened visual observation with the vector observation
        combined_obs = np.concatenate((vis_obs, vec_obs))

        # Update the obs dictionary for the agent
        flattened_obs[agent_id] = combined_obs
    return flattened_obs


def get_unique_output_dir(base_dir):
    """
    Returns a unique directory name by adding a numerical suffix if the base_dir already exists.
    """
    if not os.path.exists(base_dir):
        return base_dir
    suffix = 1
    while os.path.exists(f"{base_dir}_{suffix}"):
        suffix += 1
    return f"{base_dir}_{suffix}"


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
            no_graphics=True,
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

    base_dir = Path("results/train") / experiment_config["name"]
    absolute_base_dir = base_dir.resolve()  # Convert to absolute path
    absolute_base_dir_str = (
        absolute_base_dir.as_posix()
    )  # Convert to POSIX format (optional)

    output_dir = get_unique_output_dir(absolute_base_dir_str)

    # Create the unique directory
    os.makedirs(output_dir, exist_ok=True)

    # Save the experiment_config to the output directory
    with open(os.path.join(output_dir, "experiment_config.json"), "w") as f:
        json.dump(experiment_config, f, indent=4)

    # Run the experiment
    results = tune.Tuner(
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

    return results, output_dir.split("/")[-1]


def get_2_steps_down(start_path, steps=2):
    path = Path(start_path)
    for _ in range(steps):
        subdirs = [x for x in path.iterdir() if x.is_dir()]
        if not subdirs:  # If no more subdirectories, stop
            break
        path = subdirs[0]  # Move to the first subdirectory
    return path


# Evaluate the trained policy using the best checkpoint
def evaluate_trained_policy(
    train_results_name, checkpoint, experiment_config, config, args, eval_episodes=100
):
    experiment_config["intervention_type"] = "none"
    config["intervention_type"] = "none"

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
            no_graphics=True,
            episode_horizon=c["episode_horizon"],
            side_channels=[
                env_params_channel,
                stats_channel,
                human_intervention_channel,
            ],
        ),
    )

    # Create an evaluation environment
    eval_env = Unity3DEnv(
        run_config=experiment_config,
        file_name=args.file_name,
        no_graphics=True,
        episode_horizon=args.horizon,
        side_channels=[
            env_params_channel,
            stats_channel,
            human_intervention_channel,
        ],
    )

    # Load the trained model
    agent = PPO(config)
    agent.restore(checkpoint.checkpoint)

    base_dir = Path("results/test") / train_results_name
    absolute_base_dir = base_dir.resolve()  # Convert to absolute path
    absolute_base_dir_str = (
        absolute_base_dir.as_posix()
    )  # Convert to POSIX format (optional)
    writer = SummaryWriter(absolute_base_dir_str)

    # Save the experiment_config to the output directory
    with open(os.path.join(absolute_base_dir_str, "experiment_config.json"), "w") as f:
        json.dump(experiment_config, f, indent=4)

    total_rewards = []
    # Run evaluation episodes
    for episode in range(eval_episodes):
        obs = eval_env.reset()[0]
        flattened_obs = flatten_obs(obs)

        done = False
        total_reward = 0
        while not done:
            # self.get_policy(policy_id) and call compute_actions() on it directly.
            action = agent.compute_actions(
                flattened_obs, policy_id=list(agent.config.policies.keys())[0]
            )

            obs, rewards, terminateds, truncateds, infos = eval_env.step(action)
            flattened_obs = flatten_obs(obs)

            total_reward += sum(rewards.values())

            if all(terminateds.values()):
                done = True

        total_rewards.append(total_reward)

        # Log total reward and mean reward to TensorBoard
        writer.add_scalar("Episode Total Reward", total_reward, episode + 1)
        writer.add_scalar("Episode Mean Reward", total_reward / 3, episode + 1)

        # Example of logging side-channel metrics if available (you can customize this)
        side_channel_metrics = stats_channel.get_and_reset_stats()
        for key, value in side_channel_metrics.items():
            writer.add_scalar(
                f"SideChannel/{key}", calculate_average(value), episode + 1
            )

        print(
            f"episode {episode + 1} - episode total reward: {total_reward} - episode mean reward: {total_reward / 3} - mean reward over all episodes: {(sum(total_rewards) / (episode + 1)):0.3f}"
        )

    # Clean up
    eval_env.close()
    writer.close()


if __name__ == "__main__":

    experiment_config_dirs = [
        ### no intervention
        "src/configs/training_config_no_intervention.yml",
        ### rule based controller
        "src/configs/training_config_rule_based_llama_3.1_8b_instruct.yml",
        "src/configs/training_config_rule_based_pharia_1_7b_control_aligned.yml",
        ### natural language controller
        "src/configs/training_config_natural_language_pharia_1_7b_control_aligned.yml",
        "src/configs/training_config_natural_language_llama_3.1_8b_instruct.yml",
    ]

    for experiment_config_dir in experiment_config_dirs:
        for i in range(10):
            ray.init(local_mode=False)

            with open(experiment_config_dir, "r") as file:
                experiment_config = yaml.safe_load(file)

            args = parser.parse_args()
            # Get policies (different agent types; "behaviors" in MLAgents) and
            # the mappings from individual agents to Policies.
            policies, policy_mapping_fn = Unity3DEnv.get_policy_configs_for_game(
                args.env
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
                .framework("torch")  # "tf" if args.env != "Pyramids" else "torch"
                # For running in editor, force to use just one Worker (we only have
                # one Unity running)!
                .env_runners(
                    num_env_runners=args.num_workers if args.file_name else 0,
                    rollout_fragment_length=300,
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
                .resources(num_gpus=1.0)
                .env_runners(num_gpus_per_env_runner=1.0)
                .callbacks(CustomMetricsCallback)
            )

            config["intervention_type"] = experiment_config["intervention_type"]
            results, train_results_name = train_policy(
                experiment_config=experiment_config, config=config, args=args
            )

            # checkpoint = results.get_best_result()
            checkpoint = results.get_best_result(
                metric="AerialWildfireSuppression/Extinguishing Trees Reward_mean",
                mode="max",
            )

            evaluate_trained_policy(
                train_results_name=train_results_name,
                checkpoint=checkpoint,
                experiment_config=experiment_config,
                config=config,
                args=args,
                eval_episodes=100,
            )

            ray.shutdown()
