import os
import tensorflow as tf
import numpy as np
from collections import defaultdict


# Function to parse event files and extract metrics
def extract_metrics_from_events(event_file):
    metrics = defaultdict(list)

    for event in tf.compat.v1.train.summary_iterator(event_file):
        for value in event.summary.value:
            # Store the metric values by tag name
            metrics[value.tag].append(value.simple_value)

    return metrics


test_results = {
    "no_intervention": {
        "results_path": "C:/Users/pdsie/Documents/human_intervention_marl/results/test/no_intervention",
        "metrics": defaultdict(list),
    },
    "rule_based_llama_3.1_8b_instruct": {
        "results_path": "C:/Users/pdsie/Documents/human_intervention_marl/results/test/rule_based_llama_3.1_8b_instruct",
        "metrics": defaultdict(list),
    },
    "rule_based_pharia_1_7b_control_aligned": {
        "results_path": "C:/Users/pdsie/Documents/human_intervention_marl/results/test/rule_based_pharia_1_7b_control_aligned",
        "metrics": defaultdict(list),
    },
}

# Process each setup
for experiment_setup, path_and_metrics in test_results.items():
    # Iterate through the folders of runs
    for run_folder in os.listdir(path_and_metrics["results_path"]):
        run_path = os.path.join(path_and_metrics["results_path"], run_folder)

        # Look for the event file in the run folder
        for file_name in os.listdir(run_path):
            if file_name.startswith("events.out.tfevents"):
                event_file = os.path.join(run_path, file_name)
                # Extract metrics from this run
                run_metrics = extract_metrics_from_events(event_file)

                # Append run metrics to setup metrics
                for metric_name, values in run_metrics.items():
                    path_and_metrics["metrics"][metric_name].extend(values)

    # ANSI escape codes for different colors
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    # Print text with only the {experiment_setup} part colored
    print(f"\nAverage metrics for {CYAN}{experiment_setup}{RESET}:")
    for metric, values in path_and_metrics["metrics"].items():
        # Check if the metric is "SideChannel/AerialWildfireSuppression/Extinguishing Trees Reward"
        if metric == "SideChannel/AerialWildfireSuppression/Extinguishing Trees Reward":
            # Print the line in yellow
            print(
                f"{YELLOW}{metric:<67}mean: {np.mean(values):<12f} std: {np.std(values):<12f}{RESET}"
            )
        else:
            # Print the line in the default color
            print(
                "{:<67}mean: {:<12f} std: {:<12f}".format(
                    metric, np.mean(values), np.std(values)
                )
            )
