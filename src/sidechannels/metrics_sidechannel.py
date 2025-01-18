from ray.rllib.algorithms.callbacks import DefaultCallbacks
import yaml
from pathlib import Path

CYAN = "\033[96m"
RESET = "\033[0m"

root_path = Path(__file__).parent.parent.parent / "results/train/"


class CustomMetricsCallback(DefaultCallbacks):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_steps = 0
        self.cumulative_rewards = 0  # Track total rewards across steps in the interval
        self.all_cumulative_rewards = (
            []
        )  # Track total rewards across steps in the interval
        self.steps_in_interval = (
            0  # Track the number of steps in the 9000-step interval
        )

        self.agents_rewards_episode = {}
        self.episode = 0

    def on_episode_step(
        self,
        *,
        worker,
        base_env,
        policies=None,
        episode,
        env_index=None,
        **kwargs,
    ):

        # Handle custom side-channel logging (same as before)
        # unity_env = base_env.get_sub_environments()[0]
        # stats_side_channel = unity_env.stats_channel.stats

        # if any(
        #     [
        #         v[0] > 0
        #         for v in stats_side_channel[
        #             "AerialWildfireSuppression/Fire too Close to City"
        #         ]
        #     ]
        # ):
        #     print("")

        # Increment step counter based on the episode's current step count
        self.total_steps += episode.active_agent_steps
        self.steps_in_interval += 1

        # Get the rewards for this step and accumulate them
        agent_count = len(episode.agent_rewards)
        step_reward_sum = sum(
            [value for key, value in list(episode.agent_rewards.items())]
        )

        for agent_env_id, reward in episode.agent_rewards.items():
            agent_id, _ = agent_env_id
            if agent_id not in self.agents_rewards_episode:
                self.agents_rewards_episode[agent_id] = []

            self.agents_rewards_episode[agent_id].append(float(reward))

        # Add the step rewards to the cumulative total for the interval
        self.cumulative_rewards += step_reward_sum
        self.all_cumulative_rewards.append(step_reward_sum / agent_count)

        # Check if the total steps reach 9000-step intervals
        if self.total_steps % 9000 == 0 and self.total_steps > 0:
            # Calculate the average reward across all steps in this interval
            if self.steps_in_interval > 0:
                avg_reward = self.cumulative_rewards / (
                    self.steps_in_interval * agent_count
                )
                overall_mean_reward = sum(self.all_cumulative_rewards) / len(
                    self.all_cumulative_rewards
                )
                print(
                    f"{CYAN}steps: {self.total_steps} - mean reward over last 9000 steps: {avg_reward:0.3f} "
                    f"- mean reward over all steps: {overall_mean_reward:0.3f}{RESET}"
                )

            # Reset the cumulative rewards and step count for the next interval
            self.cumulative_rewards = 0
            self.steps_in_interval = 0

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        # Handle custom side-channel logging (same as before)
        unity_env = base_env.get_sub_environments()[0]
        stats_side_channel = unity_env.stats_channel.stats

        # if any(
        #     [
        #         v[0] > 0
        #         for v in stats_side_channel[
        #             "AerialWildfireSuppression/Fire too Close to City"
        #         ]
        #     ]
        # ):
        #     print("")

        for key, metric in stats_side_channel.items():
            episode.custom_metrics[key] = sum([value[0] for value in metric]) / len(
                metric
            )

        # add task count to metrics
        task_count = unity_env.task_count
        episode.custom_metrics["task_count"] = task_count

        # add total task count to metrics
        total_task_count = unity_env.total_task_count
        episode.custom_metrics["total_task_count"] = total_task_count

        self.agents_rewards_episode = {
            f"episode_{self.episode}": self.agents_rewards_episode
        }

        latest_modified_folder = find_latest_modified_folder(root_path)

        # Example path to the latest modified folder (from the previous function)
        file_path = latest_modified_folder / "agents_rewards_episode.yaml"

        # Ensure the directory exists (though in this case, latest_modified_folder should exist)
        latest_modified_folder.mkdir(parents=True, exist_ok=True)

        # Dump self.agents_rewards_episode to a file
        with open(file_path, "a") as f:
            yaml.dump(self.agents_rewards_episode, f, default_flow_style=None)

        # Reset the self.agents_rewards_episode to an empty dictionary
        self.agents_rewards_episode = {}

        self.episode += 1


def find_latest_modified_folder(root):
    latest_time = None
    latest_folder = None

    # Traverse the folder structure recursively
    for folder in root.glob("**/"):  # '**/' is used to look into subfolders
        if folder.is_dir():  # Only consider directories
            # Check the latest modification time of the folder itself
            folder_mtime = folder.stat().st_mtime

            # Now, check the latest modification time of the contents (files and subfolders)
            for content in folder.iterdir():
                content_mtime = content.stat().st_mtime
                # Compare both folder and content modification times
                if latest_time is None or content_mtime > latest_time:
                    latest_time = content_mtime
                    latest_folder = folder

    return latest_folder
