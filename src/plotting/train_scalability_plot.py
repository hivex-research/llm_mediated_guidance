import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

experiments = [
    {
        "episode_reward_mean": {
            "title": "Rule-Based Controller: Episode Reward Mean",
            "file_name": "baseline_vs_rb_episode_reward_mean_scalability.pdf",
            "colors": ["#cc3300", "#0072B2", "#ffcc00"],
            "y_lim_max": 1000,
            "selected_folders": [
                "NO_INTERVENTION",
                "NO_INTERVENTION_4",
                "NO_INTERVENTION_5",
                "NO_INTERVENTION_6",
                "RB_LLAMA_3.1",
                "RB_LLAMA_3.1_4",
                "RB_LLAMA_3.1_5",
                "RB_LLAMA_3.1_6",
                "RB_PHARIA_1",
                "RB_PHARIA_1_4",
                "RB_PHARIA_1_5",
                "RB_PHARIA_1_6",
            ],
        }
    },
    {
        "AerialWildfireSuppression/Extinguishing Trees Reward_mean": {
            "title": "Rule-Based Controller: Extinguishing Trees Reward Mean",
            "file_name": "baseline_vs_rb_extinguishing_trees_reward_mean_scalability.pdf",
            "colors": ["#cc3300", "#0072B2", "#ffcc00"],
            "y_lim_max": 22,
            "selected_folders": [
                "NO_INTERVENTION",
                "NO_INTERVENTION_4",
                "NO_INTERVENTION_5",
                "NO_INTERVENTION_6",
                "RB_LLAMA_3.1",
                "RB_LLAMA_3.1_4",
                "RB_LLAMA_3.1_5",
                "RB_LLAMA_3.1_6",
                "RB_PHARIA_1",
                "RB_PHARIA_1_4",
                "RB_PHARIA_1_5",
                "RB_PHARIA_1_6",
            ],
        }
    },
]

custom_legend_names = [
    "No Intervention",
    "LLAMA 3.1 Intervention",
    "PHARIA 1 Intervention",
]


# Define the root directory
root_dir = "./results/train"

# Process experiments for plotting scalability
for experiment in experiments:
    key = list(experiment.keys())[0]

    # Dictionary to store accumulated results for each folder by trial
    accumulated_data = defaultdict(lambda: defaultdict(list))

    # Traverse only the selected folders
    for folder in experiment[key]["selected_folders"]:
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_path}")  # Debugging folder processing
            # Traverse all runs (subdirectories) within the folder
            for subdir, _, files in os.walk(folder_path):
                for file in files:
                    if file == "result.json":
                        file_path = os.path.join(subdir, file)
                        print(f"Found file: {file_path}")  # Debugging file presence
                        # Open and read the JSON file
                        with open(file_path, "r") as f:
                            for i, line in enumerate(f):
                                results = json.loads(line)

                                if key == "episode_reward_mean":
                                    if key in results["env_runners"]:
                                        reward_mean = results["env_runners"][key]
                                elif (
                                    key
                                    == "AerialWildfireSuppression/Extinguishing Trees Reward_mean"
                                ):
                                    if key in results["env_runners"]["custom_metrics"]:
                                        reward_mean = results["env_runners"][
                                            "custom_metrics"
                                        ][key]
                                accumulated_data[folder][i].append(reward_mean)
        else:
            print(f"Folder not found: {folder_path}")  # Debugging missing folders

    # Create plot for scalability analysis
    agent_counts = [3, 4, 5, 6]
    folder_groups = [
        experiment[key]["selected_folders"][i : i + 4]
        for i in range(0, len(experiment[key]["selected_folders"]), 4)
    ]

    plt.figure(figsize=(10, 3))
    colors = experiment[key]["colors"]

    for idx, group in enumerate(folder_groups):
        last_5_means = []
        last_5_sems = []

        for folder in group:
            if folder in accumulated_data:
                data_per_time_step = accumulated_data[folder]
                time_steps = sorted(data_per_time_step.keys())

                # Stack the lists into an array where rows are trials and columns are values per step
                data_matrix = [data_per_time_step[trial] for trial in time_steps]
                df = pd.DataFrame(data_matrix)

                # Aggregate data across multiple runs
                mean_values = df.mean(axis=1)
                std_values = df.std(axis=1)
                n_trials = df.shape[1]  # Number of runs
                sem_values = std_values / np.sqrt(n_trials)

                # Calculate last 5 mean and SEM values
                last_5_mean = mean_values.values[-5:].mean()
                last_5_sem = sem_values.values[-5:].mean()

                last_5_means.append(last_5_mean)
                last_5_sems.append(last_5_sem)

                print(
                    f"Folder {folder}, Last 5 Mean: {last_5_mean}, SEM: {last_5_sem}"
                )  # Debugging mean and SEM values
            else:
                print(f"No data for folder: {folder}")  # Debugging missing data
                last_5_means.append(np.nan)  # Handle missing data gracefully
                last_5_sems.append(np.nan)

        # Plot scalability results for the current group
        plt.errorbar(
            agent_counts,
            last_5_means,
            yerr=last_5_sems,
            fmt="o",
            linestyle="-",
            color=colors[idx],
            label=custom_legend_names[idx],
            capsize=5,
        )

    # Label the plot
    plt.title(experiment[key]["title"])
    plt.xlabel("Number of Agents")
    plt.ylabel("Mean Reward")
    plt.xticks(agent_counts)
    plt.ylim(0, experiment[key]["y_lim_max"])
    plt.grid(True)
    plt.legend()

    # Save the plot as PDF
    plt.savefig(
        experiment[key]["file_name"],
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()
