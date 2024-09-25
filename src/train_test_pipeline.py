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


# Evaluate the trained policy using the best checkpoint
def evaluate_trained_policy(
    train_results_name, checkpoint, experiment_config, config, args, eval_episodes=100
):
    env_params_channel = EnvironmentParametersChannel()
    for key, env_parameter in experiment_config["env_parameters"].items():
        env_params_channel.set_float_parameter(key, env_parameter)
    stats_channel = StatsSideChannel()
    human_intervention_channel = HumanCommandChannel()

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
    agent.restore(checkpoint)

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
    ray.init(local_mode=True)

    # NN only
    # experiment_config_dir = "src/configs/training_config_no_intervention.yml"
    # auto
    # experiment_config_dir = "src/configs/training_config_rule_based.yml"
    # llm
    experiment_config_dirs = [
        "src/configs/training_config_no_intervention.yml",
        # "src/configs/training_config_human_llm_intervention.yml",
        # "src/configs/training_config_rule_based.yml",
        # "src/configs/training_config_human_llm_intervention_1.yml",
    ]

    for experiment_config_dir in experiment_config_dirs:

        with open(experiment_config_dir, "r") as file:
            experiment_config = yaml.safe_load(file)

        args = parser.parse_args()
        # Get policies (different agent types; "behaviors" in MLAgents) and
        # the mappings from individual agents to Policies.
        policies, policy_mapping_fn = Unity3DEnv.get_policy_configs_for_game(args.env)
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
                # entropy_coeff=0.05,
                model={"fcnet_hiddens": [256, 256]},
                # model={
                #     # Define convolutional layers
                #     "conv_filters": [
                #         [16, [8, 8], 4],  # 16 filters of size 8x8 with stride of 4
                #         [32, [4, 4], 2],  # 32 filters of size 4x4 with stride of 2
                #         [64, [3, 3], 1],  # 64 filters of size 3x3 with stride of 1
                #     ],
                #     "conv_activation": "relu",  # Activation function for conv layers
                #     "post_fcnet_hiddens": [
                #         256,
                #         256,
                #     ],  # Fully connected layers after conv layers
                # },
            )
            .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            .resources(num_gpus=1.0)
            .env_runners(num_gpus_per_env_runner=1.0)  # num_cpus_per_env_runner
            .callbacks(CustomMetricsCallback)
        )

        config["intervention_type"] = experiment_config["intervention_type"]

        train_results = {
            # "AWS_NO_INTERVENTION": "C:/Users/pdsie/Documents/human_intervention_marl/results/train/AWS_NO_INTERVENTION/PPO_2024-09-20_22-35-57/PPO_unity3d_f0839_00000_0_2024-09-20_22-35-57/checkpoint_000022",
            "AWS_NO_INTERVENTION_1": "C:/Users/pdsie/Documents/human_intervention_marl/results/train/AWS_NO_INTERVENTION_1/PPO_2024-09-20_22-39-42/PPO_unity3d_76fe5_00000_0_2024-09-20_22-39-42/checkpoint_000022",
            "AWS_NO_INTERVENTION_2": "C:/Users/pdsie/Documents/human_intervention_marl/results/train/AWS_NO_INTERVENTION_2/PPO_2024-09-20_22-41-38/PPO_unity3d_bc1ac_00000_0_2024-09-20_22-41-38/checkpoint_000022",
            "AWS_NO_INTERVENTION_3": "C:/Users/pdsie/Documents/human_intervention_marl/results/train/AWS_NO_INTERVENTION_3/PPO_2024-09-21_08-44-46/PPO_unity3d_fdc89_00000_0_2024-09-21_08-44-46/checkpoint_000022",
            "AWS_NO_INTERVENTION_4": "C:/Users/pdsie/Documents/human_intervention_marl/results/train/AWS_NO_INTERVENTION_4/PPO_2024-09-21_08-46-45/PPO_unity3d_44c14_00000_0_2024-09-21_08-46-45/checkpoint_000022",
            "AWS_NO_INTERVENTION_5": "C:/Users/pdsie/Documents/human_intervention_marl/results/train/AWS_NO_INTERVENTION_5/PPO_2024-09-21_08-59-54/PPO_unity3d_1b375_00000_0_2024-09-21_08-59-55/checkpoint_000022",
            "AWS_NO_INTERVENTION_6": "C:/Users/pdsie/Documents/human_intervention_marl/results/train/AWS_NO_INTERVENTION_6/PPO_2024-09-21_09-13-11/PPO_unity3d_f6015_00000_0_2024-09-21_09-13-11/checkpoint_000022",
            "AWS_NO_INTERVENTION_7": "C:/Users/pdsie/Documents/human_intervention_marl/results/train/AWS_NO_INTERVENTION_7/PPO_2024-09-21_18-14-45/PPO_unity3d_9df32_00000_0_2024-09-21_18-14-45/checkpoint_000022",
            "AWS_NO_INTERVENTION_8": "C:/Users/pdsie/Documents/human_intervention_marl/results/train/AWS_NO_INTERVENTION_8/PPO_2024-09-21_18-15-34/PPO_unity3d_bb57c_00000_0_2024-09-21_18-15-34/checkpoint_000022",
            "AWS_NO_INTERVENTION_9": "C:/Users/pdsie/Documents/human_intervention_marl/results/train/AWS_NO_INTERVENTION_9/PPO_2024-09-21_18-16-30/PPO_unity3d_dcb09_00000_0_2024-09-21_18-16-30/checkpoint_000022",
            "AWS_RULE_BASED_LLAMA_3_8B_INSTRUCT_INTERVENTION": "C:/Users/pdsie/Documents/human_intervention_marl/results/train/AWS_RULE_BASED_LLAMA_3_8B_INSTRUCT_INTERVENTION/PPO_2024-09-22_00-08-52/PPO_unity3d_15db6_00000_0_2024-09-22_00-08-52/checkpoint_000022",
            "AWS_RULE_BASED_LLAMA_3_8B_INSTRUCT_INTERVENTION_1": "C:/Users/pdsie/Documents/human_intervention_marl/results/train/AWS_RULE_BASED_LLAMA_3_8B_INSTRUCT_INTERVENTION_1/PPO_2024-09-22_00-17-56/PPO_unity3d_5a63b_00000_0_2024-09-22_00-17-56/checkpoint_000022",
            "AWS_RULE_BASED_LLAMA_3_8B_INSTRUCT_INTERVENTION_2": "C:/Users/pdsie/Documents/human_intervention_marl/results/train/AWS_RULE_BASED_LLAMA_3_8B_INSTRUCT_INTERVENTION_2/PPO_2024-09-22_09-30-07/PPO_unity3d_7e129_00000_0_2024-09-22_09-30-07/checkpoint_000022",
            "AWS_RULE_BASED_LLAMA_3_8B_INSTRUCT_INTERVENTION_3": "C:/Users/pdsie/Documents/human_intervention_marl/results/train/AWS_RULE_BASED_LLAMA_3_8B_INSTRUCT_INTERVENTION_3/PPO_2024-09-22_09-30-36/PPO_unity3d_8f5ea_00000_0_2024-09-22_09-30-36/checkpoint_000022",
            "AWS_RULE_BASED_LLAMA_3_8B_INSTRUCT_INTERVENTION_4": "C:/Users/pdsie/Documents/human_intervention_marl/results/train/AWS_RULE_BASED_LLAMA_3_8B_INSTRUCT_INTERVENTION_4/PPO_2024-09-22_14-09-42/PPO_unity3d_8cb51_00000_0_2024-09-22_14-09-42/checkpoint_000022",
            "AWS_RULE_BASED_LLAMA_3_8B_INSTRUCT_INTERVENTION_5": "C:/Users/pdsie/Documents/human_intervention_marl/results/train/AWS_RULE_BASED_LLAMA_3_8B_INSTRUCT_INTERVENTION_5/PPO_2024-09-22_14-11-15/PPO_unity3d_c4445_00000_0_2024-09-22_14-11-15/checkpoint_000022",
            "AWS_RULE_BASED_LLAMA_3_8B_INSTRUCT_INTERVENTION_6": "C:/Users/pdsie/Documents/human_intervention_marl/results/train/AWS_RULE_BASED_LLAMA_3_8B_INSTRUCT_INTERVENTION_6/PPO_2024-09-22_17-12-14/PPO_unity3d_0c6d9_00000_0_2024-09-22_17-12-14/checkpoint_000022",
            "AWS_RULE_BASED_LLAMA_3_8B_INSTRUCT_INTERVENTION_7": "C:/Users/pdsie/Documents/human_intervention_marl/results/train/AWS_RULE_BASED_LLAMA_3_8B_INSTRUCT_INTERVENTION_7/PPO_2024-09-22_17-39-33/PPO_unity3d_dd737_00000_0_2024-09-22_17-39-33/checkpoint_000022",
            "AWS_RULE_BASED_LLAMA_3_8B_INSTRUCT_INTERVENTION_8": "C:/Users/pdsie/Documents/human_intervention_marl/results/train/AWS_RULE_BASED_LLAMA_3_8B_INSTRUCT_INTERVENTION_8/PPO_2024-09-22_20-29-33/PPO_unity3d_9d12b_00000_0_2024-09-22_20-29-33/checkpoint_000022",
            "AWS_RULE_BASED_LLAMA_3_8B_INSTRUCT_INTERVENTION_9": "C:/Users/pdsie/Documents/human_intervention_marl/results/train/AWS_RULE_BASED_LLAMA_3_8B_INSTRUCT_INTERVENTION_9/PPO_2024-09-22_22-10-55/PPO_unity3d_c66a6_00000_0_2024-09-22_22-10-55/checkpoint_000022",
            "AWS_RULE_BASED_PHARIA_1_7B_INTERVENTION": "C:/Users/pdsie/Documents/human_intervention_marl/results/train/AWS_RULE_BASED_PHARIA_1_7B_INTERVENTION/PPO_2024-09-23_05-36-13/PPO_unity3d_fb3b6_00000_0_2024-09-23_05-36-13/checkpoint_000022",
            "AWS_RULE_BASED_PHARIA_1_7B_INTERVENTION_1": "C:/Users/pdsie/Documents/human_intervention_marl/results/train/AWS_RULE_BASED_PHARIA_1_7B_INTERVENTION_1/PPO_2024-09-23_09-03-19/PPO_unity3d_ea314_00000_0_2024-09-23_09-03-19/checkpoint_000022",
            "AWS_RULE_BASED_PHARIA_1_7B_INTERVENTION_2": "C:/Users/pdsie/Documents/human_intervention_marl/results/train/AWS_RULE_BASED_PHARIA_1_7B_INTERVENTION_2/PPO_2024-09-23_12-24-01/PPO_unity3d_f3a28_00000_0_2024-09-23_12-24-01/checkpoint_000022",
            "AWS_RULE_BASED_PHARIA_1_7B_INTERVENTION_3": "C:/Users/pdsie/Documents/human_intervention_marl/results/train/AWS_RULE_BASED_PHARIA_1_7B_INTERVENTION_3/PPO_2024-09-23_15-39-59/PPO_unity3d_53d81_00000_0_2024-09-23_15-39-59/checkpoint_000022",
            "AWS_RULE_BASED_PHARIA_1_7B_INTERVENTION_4": "C:/Users/pdsie/Documents/human_intervention_marl/results/train/AWS_RULE_BASED_PHARIA_1_7B_INTERVENTION_4/PPO_2024-09-23_19-09-01/PPO_unity3d_87ae3_00000_0_2024-09-23_19-09-01/checkpoint_000022",
            "AWS_RULE_BASED_PHARIA_1_7B_INTERVENTION_5": "C:/Users/pdsie/Documents/human_intervention_marl/results/train/AWS_RULE_BASED_PHARIA_1_7B_INTERVENTION_5/PPO_2024-09-23_22-34-36/PPO_unity3d_3fde7_00000_0_2024-09-23_22-34-36/checkpoint_000022",
            "AWS_RULE_BASED_PHARIA_1_7B_INTERVENTION_6": "C:/Users/pdsie/Documents/human_intervention_marl/results/train/AWS_RULE_BASED_PHARIA_1_7B_INTERVENTION_6/PPO_2024-09-24_02-09-25/PPO_unity3d_41e7a_00000_0_2024-09-24_02-09-25/checkpoint_000022",
            "AWS_RULE_BASED_PHARIA_1_7B_INTERVENTION_7": "C:/Users/pdsie/Documents/human_intervention_marl/results/train/AWS_RULE_BASED_PHARIA_1_7B_INTERVENTION_7/PPO_2024-09-24_05-38-00/PPO_unity3d_6570d_00000_0_2024-09-24_05-38-00/checkpoint_000022",
            "AWS_RULE_BASED_PHARIA_1_7B_INTERVENTION_8": "C:/Users/pdsie/Documents/human_intervention_marl/results/train/AWS_RULE_BASED_PHARIA_1_7B_INTERVENTION_8/PPO_2024-09-24_09-09-18/PPO_unity3d_ea348_00000_0_2024-09-24_09-09-18/checkpoint_000022",
            "AWS_RULE_BASED_PHARIA_1_7B_INTERVENTION_9": "C:/Users/pdsie/Documents/human_intervention_marl/results/train/AWS_RULE_BASED_PHARIA_1_7B_INTERVENTION_9/PPO_2024-09-24_13-00-43/PPO_unity3d_3e5ba_00000_0_2024-09-24_13-00-43/checkpoint_000022",
        }

        for k, v in train_results.items():
            # results, train_results_name = train_policy(
            #     experiment_config=experiment_config, config=config, args=args
            # )

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

            train_results_name = k

            eval_experiment_config = experiment_config.copy()
            eval_experiment_config["intervention_type"] = "none"
            eval_config = config.copy()
            eval_config["intervention_type"] = "none"

            # Load the best checkpoint
            # checkpoint = results.get_best_result().checkpoint
            checkpoint = v

            # Evaluate the trained policy after training
            evaluate_trained_policy(
                train_results_name=train_results_name,
                checkpoint=checkpoint,
                experiment_config=eval_experiment_config,
                config=eval_config,
                args=args,
                eval_episodes=100,
            )

        # for i in range(1):
        #     # results, train_results_name = train_policy(
        #     #     experiment_config=experiment_config, config=config, args=args
        #     # )

        #     # Set up channels
        #     env_params_channel = EnvironmentParametersChannel()
        #     for key, env_parameter in experiment_config["env_parameters"].items():
        #         env_params_channel.set_float_parameter(key, env_parameter)
        #     stats_channel = StatsSideChannel()
        #     human_intervention_channel = HumanCommandChannel()

        #     # Register the environment
        #     tune.register_env(
        #         "unity3d",
        #         lambda c: Unity3DEnv(
        #             run_config=experiment_config,
        #             file_name=c["file_name"],
        #             no_graphics=True,
        #             episode_horizon=c["episode_horizon"],
        #             side_channels=[
        #                 env_params_channel,
        #                 stats_channel,
        #                 human_intervention_channel,
        #             ],
        #         ),
        #     )

        #     train_results_name = "AWS_NO_INTERVENTION"

        #     eval_experiment_config = experiment_config.copy()
        #     eval_experiment_config["intervention_type"] = "none"
        #     eval_config = config.copy()
        #     eval_config["intervention_type"] = "none"

        #     # Load the best checkpoint
        #     # checkpoint = results.get_best_result().checkpoint
        #     checkpoint =

        #     # Evaluate the trained policy after training
        #     evaluate_trained_policy(
        #         train_results_name=train_results_name,
        #         checkpoint=checkpoint,
        #         experiment_config=eval_experiment_config,
        #         config=eval_config,
        #         args=args,
        #         eval_episodes=100,
        #     )

        # Load model after training
        # last_checkpoint_dir = results.get_best_result().checkpoint
        # algo = PPO.from_checkpoint(last_checkpoint_dir)

        # And check the results.
        # if args.as_test:
        #     check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
