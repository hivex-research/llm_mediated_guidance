"""Intervention-free evaluation of trained checkpoints.

Loads each final checkpoint and rolls it out with `intervention_type="none"`,
so the learned policy flies unaided. This isolates what the policy internalised
from what the mediator was doing for it during training.

Usage:
    python src/eval_no_intervention.py --episodes 10
    python src/eval_no_intervention.py --episodes 10 --conditions NO_INTERVENTION RB_LLAMA_3.1

Writes results/eval_no_intervention/raw_episodes.csv and summary.csv.
"""

import argparse
import csv
import json
import os
import socket
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))

from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel
from sidechannels.human_command_sidechannel import HumanCommandChannel

from environments.AWS_MultiAgentEnv import AerialWildFireSuppressionEnv


REPO = Path(__file__).parent.parent
DEFAULT_BINARY = (
    REPO / "env/Hivex_AerialWildfireSuppression_win/Hivex_AerialWildfireSuppression.exe"
)

POLICY_ID = "AerialWildfireSuppression"

# The five conditions reported in the paper (3-agent, default task).
CONDITIONS = [
    "NO_INTERVENTION",
    "RB_LLAMA_3.1",
    "RB_PHARIA_1",
    "NL_LLAMA_3.1",
    "NL_PHARIA_1",
]

# Do NOT hard-code env parameters here. Each archived condition stores the
# exact config it was trained with in experiment_config.json, and they differ
# between conditions (notably `no_graphics`). Injecting anything else changes
# the environment out from under the checkpoint and the numbers become
# meaningless -- evaluate each policy under the conditions it was trained in.

# results/archive/00/train is the run set the paper reports: it is the only
# root holding all five conditions, at exactly 10 complete runs each. The other
# roots contain partial duplicates and re-runs, so mixing them in would evaluate
# a different set of seeds than Table 1 was computed from.
PAPER_ROOT = REPO / "results/archive/00/train"

OTHER_ROOTS = [
    REPO / "results/train",
    REPO / "results/archive/01/train",
    REPO / "results/archive/train",
]


def condition_dirs(condition, root):
    """Directories holding a condition's runs.

    The archived conditions put all ten runs under one directory. The retrained
    baseline instead writes one directory per seed
    (NO_INTERVENTION_NOGFX_seed0..9), because each seed is a separate Tune run,
    so accept that layout too and treat the set as one condition.
    """
    exact = root / condition
    if exact.is_dir():
        return [exact]
    return sorted(d for d in root.glob(condition + "_seed*") if d.is_dir())


def find_final_checkpoints(condition, roots=None, index=-1):
    """Return one checkpoint per training run (=seed) for a condition.

    `index` selects which checkpoint along the run: -1 (default) is the final
    one; a non-negative value picks that position, which is how the learning
    curve is sampled. Runs with too few checkpoints are skipped rather than
    silently contributing an earlier one.
    """
    checkpoints = []
    for root in roots or [PAPER_ROOT]:
        for cond_dir in condition_dirs(condition, root):
            # <cond>/PPO_<date>/PPO_unity3d_<hash>/checkpoint_XXXXXX
            for trial_dir in sorted(cond_dir.glob("*/PPO_unity3d_*")):
                ckpts = sorted(trial_dir.glob("checkpoint_*"))
                ckpts = [c for c in ckpts if (c / "policies" / POLICY_ID).is_dir()]
                if not ckpts:
                    continue
                if index == -1:
                    checkpoints.append(ckpts[-1])
                elif index < len(ckpts):
                    checkpoints.append(ckpts[index])
    return checkpoints


def _port_free(port):
    """True only if the port is free the way the gRPC server will bind it.

    The communicator listens on 0.0.0.0 and [::], so probing 127.0.0.1 is not
    enough -- a loopback bind succeeds on ports that are already held on the
    wildcard addresses. SO_REUSEADDR is deliberately not set: it would let the
    probe bind over an existing listener and report a busy port as free.
    """
    for family, addr in (
        (socket.AF_INET, ("0.0.0.0", port)),
        (socket.AF_INET6, ("::", port)),
    ):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as sock:
                if family == socket.AF_INET6:
                    try:
                        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
                    except OSError:
                        pass
                sock.bind(addr)
        except OSError:
            return False
    return True


def pick_worker_id(start, limit=400):
    """Find a worker id whose Unity port is free.

    The env retries occupied ports with a random 1-10s sleep between attempts,
    so walking into a block of busy ports costs minutes. Probing first turns
    that into milliseconds.
    """
    base = AerialWildFireSuppressionEnv._BASE_PORT_ENVIRONMENT
    for wid in range(start, start + limit):
        if _port_free(base + wid):
            return wid
    raise RuntimeError(
        "no free Unity port in %d..%d -- are there stray Hivex processes?"
        % (base + start, base + start + limit)
    )


def load_train_config(condition, roots=None):
    """Read the config a condition was actually trained with."""
    for root in roots or [PAPER_ROOT]:
        for cond_dir in condition_dirs(condition, root):
            cfg = cond_dir / "experiment_config.json"
            if cfg.is_file():
                with open(cfg) as f:
                    return json.load(f)
    raise FileNotFoundError(
        "no experiment_config.json for %s; cannot reproduce its training "
        "environment" % condition
    )


def make_env(binary, horizon, seed, worker_offset, train_config, no_graphics=None):
    env_params_channel = EnvironmentParametersChannel()
    # Verbatim from the training config -- see the note above.
    for key, value in train_config.get("env_parameters", {}).items():
        env_params_channel.set_float_parameter(key, float(value))
    stats_channel = StatsSideChannel()
    human_channel = HumanCommandChannel()

    # `no_graphics=True` launches Unity with -nographics -batchmode, which
    # skips graphics-device init, so camera sensors render blank. The policy's
    # observation is 5292 visual + 8 vector values, so getting this wrong
    # destroys almost the entire observation. Mirror the training run unless
    # explicitly overridden.
    if no_graphics is None:
        no_graphics = bool(train_config.get("no_graphics", False))

    run_config = {
        # The one deliberate deviation from training: disable the mediator, so
        # no interpreter is built and every agent stays on "NN" throughout.
        "intervention_type": "none",
        "no_graphics": no_graphics,
    }

    AerialWildFireSuppressionEnv._WORKER_ID = worker_offset
    env = AerialWildFireSuppressionEnv(
        run_config=run_config,
        file_name=str(binary),
        no_graphics=no_graphics,
        seed=seed,
        episode_horizon=horizon,
        side_channels=[env_params_channel, stats_channel, human_channel],
    )
    return env, stats_channel


def read_stat(stats, name):
    """Sum a StatsSideChannel entry by exact metric name.

    Matching on a substring is unsafe here: "Extinguishing Trees" is a prefix
    of "Extinguishing Trees Reward", so a substring match can silently return
    the wrong series depending on dict ordering.
    """
    for key, values in stats.items():
        # Keys arrive as "AerialWildfireSuppression/<Metric Name>".
        if key.split("/")[-1] == name:
            return float(np.sum([v[0] for v in values]))
    return 0.0


def rollout(env, stats_channel, policy, horizon, explore=True):
    obs, _ = env.reset()
    stats_channel.get_and_reset_stats()

    per_agent_return = {aid: 0.0 for aid in obs}
    steps = 0

    while steps < horizon:
        action_dict = {}
        for agent_id, agent_obs in obs.items():
            action, _, _ = policy.compute_single_action(agent_obs, explore=explore)
            action_dict[agent_id] = action

        obs, rewards, terminateds, truncateds, _ = env.step(action_dict)

        for agent_id, reward in rewards.items():
            per_agent_return[agent_id] = per_agent_return.get(agent_id, 0.0) + reward
        steps += 1

        if terminateds.get("__all__", False) or truncateds.get("__all__", False):
            break

    stats = stats_channel.get_and_reset_stats()
    if os.environ.get("EVAL_DEBUG_STATS"):
        print("    stats keys: %s" % sorted(stats))
    return {
        "episode_reward_mean": float(np.mean(list(per_agent_return.values()))),
        "episode_reward_sum": float(np.sum(list(per_agent_return.values()))),
        "extinguishing_trees_reward": read_stat(
            stats, "Extinguishing Trees Reward"
        ),
        "extinguishing_trees": read_stat(stats, "Extinguishing Trees"),
        "crash_count": read_stat(stats, "Crash Count"),
        "steps": steps,
    }


def write_summary(raw_rows, out_dir):
    """Per-condition means with bootstrap 95% CIs over episodes."""
    summary_path = Path(out_dir) / "summary.csv"
    rng = np.random.default_rng(0)
    conditions = []
    for r in raw_rows:  # preserve encounter order, no duplicates
        if r["condition"] not in conditions:
            conditions.append(r["condition"])

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["condition", "metric", "mean", "ci_low", "ci_high", "n_episodes"]
        )
        for condition in conditions:
            rows = [r for r in raw_rows if r["condition"] == condition]
            for metric in ["episode_reward_mean", "extinguishing_trees_reward"]:
                values = np.array([float(r[metric]) for r in rows])
                boots = np.array(
                    [
                        rng.choice(values, size=len(values), replace=True).mean()
                        for _ in range(10000)
                    ]
                )
                writer.writerow(
                    [
                        condition,
                        metric,
                        f"{values.mean():.3f}",
                        f"{np.percentile(boots, 2.5):.3f}",
                        f"{np.percentile(boots, 97.5):.3f}",
                        len(values),
                    ]
                )
    return summary_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=3000)
    parser.add_argument("--file-name", type=str, default=str(DEFAULT_BINARY))
    parser.add_argument("--conditions", nargs="*", default=CONDITIONS)
    parser.add_argument(
        "--all-roots",
        action="store_true",
        help="Also search results/train and results/archive/01 (re-runs and "
        "partial duplicates), not just the run set the paper reports.",
    )
    ng = parser.add_mutually_exclusive_group()
    ng.add_argument(
        "--no-graphics",
        dest="no_graphics",
        action="store_true",
        help="Force -nographics for every condition. Camera sensors render "
        "blank in this mode, so only use it to reproduce a run that was "
        "trained that way.",
    )
    ng.add_argument(
        "--graphics",
        dest="no_graphics",
        action="store_false",
        help="Force graphics on for every condition.",
    )
    parser.set_defaults(no_graphics=None)  # default: mirror each training run
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Take the argmax action instead of sampling. Note the "
        "training metrics in the paper come from stochastic rollouts, so "
        "sampling (the default) is the like-for-like comparison; with "
        "argmax the discrete drop-water branch may never fire.",
    )
    parser.add_argument(
        "--checkpoint-indices",
        type=int,
        nargs="+",
        default=[-1],
        help="Which checkpoints along each run to evaluate. -1 (default) "
        "is the final one; give several indices to trace a learning curve "
        "of unassisted performance against training progress.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Rebuild summary.csv from an existing raw_episodes.csv "
        "without re-running the sweep.",
    )
    parser.add_argument(
        "--out", type=str, default=str(REPO / "results/eval_no_intervention")
    )
    args = parser.parse_args()

    out_dir = Path(args.out)

    # Recovery path: rebuild summary.csv from whatever episodes made it to disk,
    # without re-running anything.
    if args.summarize_only:
        raw_path = out_dir / "raw_episodes.csv"
        with open(raw_path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            print(f"{raw_path} has no episodes")
            return
        summary_path = write_summary(rows, out_dir)
        print(f"Summarised {len(rows)} episode(s) -> {summary_path}")
        return

    roots = [PAPER_ROOT] + (OTHER_ROOTS if args.all_roots else [])
    print("checkpoint roots: %s\n" % ", ".join(str(r) for r in roots))

    from ray.rllib.policy.policy import Policy

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = []
    worker_offset = 100

    # Append each episode as it completes. A sweep is 250 episodes over several
    # hours; writing only at the end means one crash or Ctrl+C loses all of it.
    raw_path = out_dir / "raw_episodes.csv"
    raw_fields = [
        "episode_reward_mean",
        "episode_reward_sum",
        "extinguishing_trees_reward",
        "extinguishing_trees",
        "crash_count",
        "steps",
        "condition",
        "checkpoint_index",
        "run",
        "episode",
        "checkpoint",
    ]
    raw_file = open(raw_path, "w", newline="")
    raw_writer = csv.DictWriter(raw_file, fieldnames=raw_fields)
    raw_writer.writeheader()
    raw_file.flush()

    for condition in args.conditions:
      for ckpt_index in args.checkpoint_indices:
        checkpoints = find_final_checkpoints(condition, roots, ckpt_index)
        if not checkpoints:
            print(f"[skip] no checkpoints found for {condition}")
            continue
        train_config = load_train_config(condition, roots)
        env_params = train_config.get("env_parameters", {})
        effective_ng = (
            args.no_graphics
            if args.no_graphics is not None
            else bool(train_config.get("no_graphics", False))
        )
        print(
            f"[{condition}] ckpt_index={ckpt_index} | {len(checkpoints)} run(s) | "
            f"no_graphics={effective_ng} | env_parameters={env_params}"
        )

        for run_idx, ckpt in enumerate(checkpoints):
            policy = Policy.from_checkpoint(str(ckpt / "policies" / POLICY_ID))

            worker_offset = pick_worker_id(worker_offset + 1)
            env, stats_channel = make_env(
                args.file_name,
                args.horizon,
                seed=run_idx,
                worker_offset=worker_offset,
                train_config=train_config,
                no_graphics=args.no_graphics,
            )
            try:
                for ep in range(args.episodes):
                    result = rollout(
                        env, stats_channel, policy, args.horizon,
                        explore=not args.deterministic,
                    )
                    result.update(
                        condition=condition,
                        checkpoint_index=ckpt_index,
                        run=run_idx,
                        episode=ep,
                        checkpoint=str(ckpt),
                    )
                    raw_rows.append(result)
                    raw_writer.writerow(result)
                    raw_file.flush()
                    print(
                        f"  run {run_idx} ep {ep}: "
                        f"R={result['episode_reward_mean']:.2f} "
                        f"ext={result['extinguishing_trees_reward']:.2f}"
                    )
            finally:
                try:
                    env.close()
                except Exception as exc:
                    print(f"warning: env.close() failed: {exc}")

    raw_file.close()

    if not raw_rows:
        print("No results collected.")
        return

    # raw_episodes.csv was written incrementally above.

    summary_path = write_summary(raw_rows, out_dir)
    print(f"\nWrote {raw_path}\nWrote {summary_path}")


if __name__ == "__main__":
    main()
