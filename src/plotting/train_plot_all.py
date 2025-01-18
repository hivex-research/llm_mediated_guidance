import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# Define the root directory
root_dir = "./results/final/train"

# Custom legend names for folders
custom_legend_names = [
    "No Intervention",
    "LLAMA 3.1 Intervention",
    "PHARIA 1 Intervention",
]

# Define folder groups for comparison
folder_groups = [
    {
        "name": "Rule-Based",
        "folders": ["NO_INTERVENTION", "RB_LLAMA_3.1", "RB_PHARIA_1"],
        "colors": ["#cc3300", "#0072B2", "#ffcc00"],
    },
    {
        "name": "Natural Language",
        "folders": ["NO_INTERVENTION", "NL_LLAMA_3.1", "NL_PHARIA_1"],
        "colors": ["#cc3300", "#0072B2", "#ffcc00"],
    },
]

ignore_keys = [
    "AerialWildfireSuppression/Agent ID_max",
    "AerialWildfireSuppression/Agent ID_min",
    "AerialWildfireSuppression/Agent ID_mean",
    "num_faulty_episodes",
]

# Step 1: Collect all available keys
all_keys = set()

for group in folder_groups:
    for folder in group["folders"]:
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            for subdir, _, files in os.walk(folder_path):
                for file in files:
                    if file == "result.json":
                        file_path = os.path.join(subdir, file)
                        with open(file_path, "r") as f:
                            for line in f:
                                results = json.loads(line)
                                # Collect keys from `env_runners` and `custom_metrics`
                                if "env_runners" in results:
                                    all_keys.update(results["env_runners"].keys())
                                    if "custom_metrics" in results["env_runners"]:
                                        all_keys.update(
                                            results["env_runners"][
                                                "custom_metrics"
                                            ].keys()
                                        )

all_keys = [key for key in all_keys if key not in ignore_keys]

# Define output directory for plots
output_dir = "./plots"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Initialize global minimum and maximum values for y-limit
global_min = float("inf")  # Start with infinity for min
global_max = float("-inf")  # Start with negative infinity for max

# Iterate over all keys
for key in all_keys:
    print(f"Processing key: {key}")
    global_min = float("inf")  # Reset global min for each key
    global_max = float("-inf")  # Reset global max for each key

    # Compute global min and max across both folder groups
    for group in folder_groups:
        selected_folders = group["folders"]

        accumulated_data = defaultdict(lambda: defaultdict(list))

        # Collect data for the current key and folder group
        for folder in selected_folders:
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                for subdir, _, files in os.walk(folder_path):
                    for file in files:
                        if file == "result.json":
                            file_path = os.path.join(subdir, file)
                            with open(file_path, "r") as f:
                                for i, line in enumerate(f):
                                    results = json.loads(line)

                                    if key in results.get("env_runners", {}):
                                        value = results["env_runners"][key]
                                    elif key in results.get("env_runners", {}).get(
                                        "custom_metrics", {}
                                    ):
                                        value = results["env_runners"][
                                            "custom_metrics"
                                        ][key]
                                    else:
                                        continue

                                    if isinstance(value, dict):
                                        if "mean" in value:
                                            value = value["mean"]
                                        else:
                                            continue

                                    accumulated_data[folder][i].append(value)

        # Calculate global min and max based on SEM-adjusted values
        for folder, data_per_time_step in accumulated_data.items():
            time_steps = sorted(data_per_time_step.keys())
            scaled_time_steps = [step * 9000 for step in time_steps]
            data_matrix = [data_per_time_step[trial] for trial in time_steps]
            df = pd.DataFrame(data_matrix)

            max_step = 300000
            scaled_time_steps = [step for step in scaled_time_steps if step <= max_step]
            df = df.iloc[: len(scaled_time_steps)]

            if df.empty:
                continue

            mean_values = df.mean(axis=1)
            std_values = df.std(axis=1)
            n_trials = df.shape[1]
            sem_values = std_values / np.sqrt(n_trials)

            # Adjust global_min and global_max using mean ± SEM
            global_min = min(global_min, (mean_values - sem_values).min())
            global_max = max(global_max, (mean_values + sem_values).max())

    # Plot data for both folder groups
    for group in folder_groups:
        group_name = group["name"]
        selected_folders = group["folders"]
        colors = group["colors"]

        accumulated_data = defaultdict(lambda: defaultdict(list))

        for folder in selected_folders:
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                for subdir, _, files in os.walk(folder_path):
                    for file in files:
                        if file == "result.json":
                            file_path = os.path.join(subdir, file)
                            with open(file_path, "r") as f:
                                for i, line in enumerate(f):
                                    results = json.loads(line)

                                    if key in results.get("env_runners", {}):
                                        value = results["env_runners"][key]
                                    elif key in results.get("env_runners", {}).get(
                                        "custom_metrics", {}
                                    ):
                                        value = results["env_runners"][
                                            "custom_metrics"
                                        ][key]
                                    else:
                                        continue

                                    if isinstance(value, dict):
                                        if "mean" in value:
                                            value = value["mean"]
                                        else:
                                            continue

                                    accumulated_data[folder][i].append(value)

        # Ensure there is data to plot
        if len(accumulated_data) == 0:
            print(f"No data found for key: {key}, group: {group_name}")
            continue

        plt.figure(figsize=(10, 3))

        for idx, (folder, data_per_time_step) in enumerate(accumulated_data.items()):
            time_steps = sorted(data_per_time_step.keys())
            scaled_time_steps = [step * 9000 for step in time_steps]
            data_matrix = [data_per_time_step[trial] for trial in time_steps]
            df = pd.DataFrame(data_matrix)

            max_step = 300000
            scaled_time_steps = [step for step in scaled_time_steps if step <= max_step]
            df = df.iloc[: len(scaled_time_steps)]

            if df.empty:
                print(f"No valid data to plot for folder: {folder}, key: {key}")
                continue

            mean_values = df.mean(axis=1)
            std_values = df.std(axis=1)
            n_trials = df.shape[1]
            sem_values = std_values / np.sqrt(n_trials)

            plt.plot(
                scaled_time_steps,
                mean_values,
                label=f"{folder}",
                color=colors[idx % len(colors)],
                linestyle="-",
                marker=None,
            )

            plt.fill_between(
                scaled_time_steps,
                mean_values - sem_values,
                mean_values + sem_values,
                color=colors[idx % len(colors)],
                alpha=0.2,
            )

        # Set y-axis limit based on global min and max
        if global_min < global_max:  # Ensure valid range
            plt.ylim(global_min, global_max)

        plt.title(f"{group_name}: Plot for {key}")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        plt.legend(loc="upper left")
        plt.grid(True)

        # Save the plot in the output directory
        file_name = f"{key.replace('/', '_').replace(' ', '_')}_{group_name}.pdf"
        file_path = os.path.join(output_dir, file_name)
        plt.savefig(file_path, format="pdf", bbox_inches="tight")
        plt.close()

        print(f"Plot saved to {file_path}")
