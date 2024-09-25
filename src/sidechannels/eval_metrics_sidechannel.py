from ray.rllib.algorithms.callbacks import DefaultCallbacks


class EvalCustomMetricsCallback(DefaultCallbacks):
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
        # Increment step counter based on the episode's current step count
        self.total_steps += 3
        self.steps_in_interval += 1

        # Get the rewards for this step and accumulate them
        agent_count = len(episode.agent_rewards)
        step_reward_sum = sum(
            [value for key, value in list(episode.agent_rewards.items())]
        )

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
                    f"steps: {self.total_steps} - mean reward over last 9000 steps: {avg_reward:0.3f} "
                    f"- mean reward over all steps: {overall_mean_reward:0.3f}"
                )

            # Reset the cumulative rewards and step count for the next interval
            self.cumulative_rewards = 0
            self.steps_in_interval = 0

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        # Handle custom side-channel logging
        unity_env = base_env.get_sub_environments()[0]
        stats_side_channel = unity_env.stats_channel.stats

        if "total reward" not in episode.hist_data:
            episode.hist_data["total reward"] = []
        if "mean reward" not in episode.hist_data:
            episode.hist_data["mean reward"] = []

        for key, metric in stats_side_channel.items():
            metric_sum = sum(
                [value[0] for value in metric[-len(episode.agent_rewards) :]]
            )
            if key not in episode.hist_data:
                episode.hist_data[key] = []
            episode.hist_data[key].append(metric_sum / len(episode.agent_rewards))

        # Log metrics to TensorBoard
        # Use the worker's logger to log custom metrics to TensorBoard
        logger = worker.policy_map["default_policy"].logger

        # Compute the total reward and mean reward
        total_reward = sum([sum(r) for r in episode.agent_rewards.values()])
        mean_reward = total_reward / len(episode.agent_rewards)

        # Log custom metrics
        logger.log_metrics(
            {
                "custom_metrics/total_reward": total_reward,
                "custom_metrics/mean_reward": mean_reward,
            }
        )

        # Log additional custom metrics from the side-channel stats
        for key, values in episode.hist_data.items():
            logger.log_metrics(
                {
                    f"custom_metrics/{key}": sum(values) / len(values),
                }
            )
