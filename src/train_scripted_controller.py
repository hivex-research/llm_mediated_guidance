"""Train the Rule-Based controller with the LLM removed (ablation).

Why this exists
---------------
The Rule-Based Controller's directive is "instruct agent(s) to go to their
closest fire", and the mediator's output for it is a 3x3 grid cell per agent.
That mapping -- nearest observed fire, then the cell containing it -- is
arithmetic, so it is not obvious that routing it through a 7B/8B language model
contributes anything. This condition performs the mapping directly
(`intervention_type: "scripted"`), holding everything else fixed, and so
isolates the LLM's contribution to the Rule-Based result.

Configuration matches the archived RB conditions exactly apart from the
controller itself, including `no_graphics: true`, so it is directly comparable
to RB_LLAMA_3.1 and RB_PHARIA_1.

Seeds run sequentially; each writes its own
results/train/SCRIPTED_CONTROLLER_seed<N>/ and is complete on its own, so the
run can be stopped at any point and whatever finished is usable.

    python src/train_scripted_controller.py --seeds 0 1 2 3 4 5 6 7 8 9
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import ray

from train_pipeline import train_policy, default_path
from train_baseline_nographics import Args, build_config

BASE_CONFIG = {
    "name": "SCRIPTED_CONTROLLER",
    "env_parameters": {"random_village_location": 0},
    "no_graphics": True,
    # The ablation: same directive, same task format, no language model.
    "intervention_type": "scripted",
    "lr": 0.005,
    "lambda_": 0.95,
    "gamma": 0.99,
    "sgd_minibatch_size": 900,
    "train_batch_size": 9000,
    "num_sgd_iter": 3,
    "clip_param": 0.2,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--file-name", type=str, default=str(default_path))
    p.add_argument("--horizon", type=int, default=3000)
    p.add_argument("--stop-timesteps", type=int, default=1_000_000)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--env", type=str, default="AerialWildfireSuppression")
    p.add_argument("--num-gpus", type=float, default=1.0)
    opts = p.parse_args()

    current_dir = os.path.abspath(os.getcwd())
    ray.init(
        local_mode=False,
        runtime_env={
            "env_vars": {"PYTHONPATH": current_dir},
            "working_dir": current_dir,
            "excludes": [".git/", "*.pyc", "__pycache__/", "env/", ".venv/", "results/"],
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
