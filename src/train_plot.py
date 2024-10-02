import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

experiments = [
    {
        "episode_reward_mean": {
            "title": "Rule-Based Controller: Episode Reward Mean",
            "file_name": "baseline_vs_rb_episode_reward_mean.pdf",
            "colors": [
                "#cc3300",
                "#ff6600",
                "#ffcc00",
            ],
            "y_lim_max": 12000,
            "selected_folders": [
                "NO_INTERVENTION",
                "RB_LLAMA_3.1",
                "RB_PHARIA_1",
            ],
        }
    },
    {
        "episode_reward_mean": {
            "title": "Natural Language Controller: Episode Reward Mean",
            "file_name": "baseline_vs_nl_episode_reward_mean.pdf",
            "colors": ["#cc3300", "#0EAB8A", "#0072B2"],
            "y_lim_max": 12000,
            "selected_folders": [
                "NO_INTERVENTION",
                "NL_LLAMA_3.1",
                "NL_PHARIA_1",
            ],
        }
    },
    {
        "AerialWildfireSuppression/Extinguishing Trees Reward_mean": {
            "title": "Rule-Based Controller: Extinguishing Trees Reward Mean",
            "file_name": "baseline_vs_rb_extinguishing_trees_reward_mean.pdf",
            "colors": [
                "#cc3300",
                "#ff6600",
                "#ffcc00",
            ],
            "y_lim_max": 800,
            "selected_folders": [
                "NO_INTERVENTION",
                "RB_LLAMA_3.1",
                "RB_PHARIA_1",
            ],
        }
    },
    {
        "AerialWildfireSuppression/Extinguishing Trees Reward_mean": {
            "title": "Natural Language Controller: Extinguishing Trees Reward Mean",
            "file_name": "baseline_vs_nl_extinguishing_trees_reward_mean.pdf",
            "colors": ["#cc3300", "#0EAB8A", "#0072B2"],
            "y_lim_max": 800,
            "selected_folders": [
                "NO_INTERVENTION",
                "NL_LLAMA_3.1",
                "NL_PHARIA_1",
            ],
        }
    },
]

# Define custom legend names for each folder
custom_legend_names = [
    "No Intervention",
    "LLAMA 3.1 Intervention",
    "PHARIA 1 Intervention",
]

# Define the root directory
root_dir = "./results/final/train"

# First pass to calculate global y-axis limits
for experiment in experiments:
    key = list(experiment.keys())[0]

    # Dictionary to store accumulated results for each folder by trial
    accumulated_data = defaultdict(lambda: defaultdict(list))

    # Traverse only the selected folders
    for folder in experiment[key]["selected_folders"]:
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            # Traverse subfolders and look for 'results.json'
            for subdir, _, files in os.walk(folder_path):
                for file in files:
                    if file == "result.json":
                        file_path = os.path.join(subdir, file)
                        # Open and read the JSON file
                        with open(file_path, "r") as f:
                            for i, line in enumerate(f):
                                results = json.loads(line)

                                if key == "episode_reward_mean":
                                    if key in results["env_runners"]:
                                        reward_mean = results["env_runners"][key]
                                        accumulated_data[folder][i].append(reward_mean)
                                elif (
                                    key
                                    == "AerialWildfireSuppression/Extinguishing Trees Reward_mean"
                                ):
                                    if key in results["env_runners"]["custom_metrics"]:
                                        reward_mean = results["env_runners"][
                                            "custom_metrics"
                                        ][key]
                                accumulated_data[folder][i].append(reward_mean)


for experiment in experiments:
    key = list(experiment.keys())[0]

    accumulated_data = defaultdict(lambda: defaultdict(list))

    # Traverse only the selected folders
    for folder in experiment[key]["selected_folders"]:
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            for subdir, _, files in os.walk(folder_path):
                for file in files:
                    if file == "result.json":
                        file_path = os.path.join(subdir, file)
                        with open(file_path, "r") as f:
                            for i, line in enumerate(f):
                                results = json.loads(line)

                                if key == "episode_reward_mean":
                                    if key in results["env_runners"]:
                                        reward_mean = results["env_runners"][key]
                                        accumulated_data[folder][i].append(reward_mean)
                                elif (
                                    key
                                    == "AerialWildfireSuppression/Extinguishing Trees Reward_mean"
                                ):
                                    if key in results["env_runners"]["custom_metrics"]:
                                        reward_mean = results["env_runners"][
                                            "custom_metrics"
                                        ][key]
                                accumulated_data[folder][i].append(reward_mean)

    # Now we have all the data, let's process and plot it
    plt.figure(figsize=(10, 3))

    colors = experiment[key]["colors"]

    # Plot each folder's data
    for idx, (folder, data_per_time_step) in enumerate(accumulated_data.items()):
        time_steps = sorted(data_per_time_step.keys())

        # Multiply time_steps by 9000 to reflect the correct scale
        scaled_time_steps = [step * 9000 for step in time_steps]

        # Stack the lists into an array where rows are trials and columns are values per step
        data_matrix = [data_per_time_step[trial] for trial in time_steps]
        df = pd.DataFrame(data_matrix)

        # Limit to the first 300,000 steps
        max_step = 300000
        scaled_time_steps = [step for step in scaled_time_steps if step <= max_step]
        df = df.iloc[: len(scaled_time_steps)]

        # Plot the mean line
        mean_values = df.mean(axis=1)
        plt.plot(
            scaled_time_steps,
            mean_values,
            label=f"{custom_legend_names[idx]}",
            color=colors[idx],
            linestyle="-",
            marker=None,
        )

        # Shaded region between min and max
        plt.fill_between(
            scaled_time_steps,
            df.min(axis=1),
            df.max(axis=1),
            color=colors[idx],
            alpha=0.2,
        )

    # Labeling the plot
    plt.title(experiment[key]["title"])
    plt.xlabel("Step")
    plt.ylabel("Reward Mean")

    # Set the global y-axis limits
    plt.ylim(0, experiment[key]["y_lim_max"])

    # Set x-axis to scientific notation
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    plt.legend(loc="upper left")
    plt.grid(True)

    plt.savefig(
        experiment[key]["file_name"],
        format="pdf",
        bbox_inches="tight",
    )  # Save the plot as PDF
