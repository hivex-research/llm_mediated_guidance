import logging
from typing import List, Type, Union
import numpy as np
from pathlib import Path

# RLlib
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch

from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import (
    explained_variance,
    sequence_mask,
    warn_if_infinite_kl_divergence,
)

from torch import Tensor

torch, nn = try_import_torch()
logger = logging.getLogger(__name__)


class HIVEXPPOTorchPolicy(PPOTorchPolicy):
    """PyTorch policy class used with PPO."""

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.intervention_type = config["intervention_type"]

    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[Tensor, List[Tensor]]:
        """Compute loss for Proximal Policy Objective.

        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.

        Returns:
            The PPO loss tensor given the input batch.
        """

        # train_actions_copy = train_batch.__getitem__("infos").clone()
        infos = train_batch.__getitem__("infos")
        # train_batch_actions = train_batch.__getitem__("actions")
        # # for batch in train_batch_actions:
        # #     torch.clamp(batch, min=-1, max=1)

        # action_values_file_path = Path(__file__) / "action_values.txt"
        # # Ensure the parent directory exists (and not mistakenly treat a file as a directory)
        # if not action_values_file_path.parent.exists():
        #     action_values_file_path.parent.mkdir(parents=True, exist_ok=True)

        # # go through all batches and dump first value in file
        # for batch in train_batch_actions:
        #     if len(batch) == 0:
        #         continue
        #     v = batch[0]
        #     if v < -1 or v > 1:
        #         # print("Action value out of bounds: ", v)
        #         # dump to file
        #         with open(action_values_file_path, "a") as f:
        #             f.write(str(v) + "\n")

        if type(infos) != torch.Tensor:
            if any(["NN" not in info for info in infos]):
                train_batch_actions = train_batch.__getitem__("actions")
                for index, info in enumerate(infos):
                    if "NN" not in info:
                        if self.intervention_type in info:
                            train_batch_actions[index] = torch.tensor(
                                [
                                    info[self.intervention_type][0][0],
                                    np.float32(info[self.intervention_type][1][0]),
                                ],
                                dtype=torch.float32,
                                device=train_batch_actions.device,  # Assume using same device
                            )

                train_batch.__setitem__("actions", train_batch_actions)

        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
            # TODO smorad: should we do anything besides warn? Could discard KL term
            # for this update
            warn_if_infinite_kl_divergence(self, mean_kl_loss)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES]
            * torch.clamp(
                logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
            ),
        )

        # Compute a value function loss.
        if self.config["use_critic"]:
            value_fn_out = model.value_function()
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        # Ignore the value function.
        else:
            value_fn_out = torch.tensor(0.0).to(surrogate_loss.device)
            vf_loss_clipped = mean_vf_loss = torch.tensor(0.0).to(surrogate_loss.device)

        total_loss = reduce_mean_valid(
            -surrogate_loss
            + self.config["vf_loss_coeff"] * vf_loss_clipped
            - self.entropy_coeff * curr_entropy
        )

        # Add mean_kl_loss (already processed through `reduce_mean_valid`),
        # if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = reduce_mean_valid(-surrogate_loss)
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss

        return total_loss
