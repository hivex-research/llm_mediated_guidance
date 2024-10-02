import numpy as np
from typing import Callable, Optional, Tuple, List
import random
import time

from intervention.intervention_interpreter import (
    InterventionInterpreter,
)

# RLlib
from ray.rllib.utils.typing import MultiAgentDict
from gymnasium.spaces import Box, Discrete, Tuple as TupleSpace
from ray.rllib.env.wrappers.unity3d_env import Unity3DEnv
from ray.rllib.env.multi_agent_env import (
    MultiAgentEnv,
)
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.typing import MultiAgentDict, PolicyID, AgentID
from ray.rllib.examples._old_api_stack.policy.random_policy import RandomPolicy


# ML-Agents
from mlagents_envs.side_channel.side_channel import SideChannel
from mlagents_envs.base_env import ActionTuple

# Hivex
from policies.hivex_ppo_torch_policy import HIVEXPPOTorchPolicy

YELLOW = "\033[93m"
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

STRATEGY_PRINT = """┏--------------------------------------------{RED}STRATEGY{RESET}-----------------------------------------------------┑
TEMPERATURE: {RED}{TEMPERATURE}{RESET}
|---------------------------------------------------------------------------------------------------------|
{STRATEGY}
┗---------------------------------------------------------------------------------------------------------┙"""

LLM_MEDIATOR_RESPONSE_PRINT = """┏------------------------------------------{GREEN}LLM-MEDIATOR{RESET}---------------------------------------------------┑
{RESPONSE}
┗---------------------------------------------------------------------------------------------------------┙"""


class AerialWildFireSuppressionEnv(MultiAgentEnv):

    # Default base port when connecting directly to the Editor
    _BASE_PORT_EDITOR = 5004
    # Default base port when connecting to a compiled environment
    _BASE_PORT_ENVIRONMENT = 5005
    # The worker_id for each environment instance
    _WORKER_ID = 0

    def __init__(
        self,
        run_config,
        file_name: str = None,
        port: Optional[int] = None,
        seed: int = 0,
        no_graphics: bool = False,
        timeout_wait: int = 300,
        episode_horizon: int = 1000,
        side_channels: List[SideChannel] = None,
        task_decay: int = 300,
    ):
        super().__init__()

        if file_name is None:
            print(
                "No game binary provided, will use a running Unity editor "
                "instead.\nMake sure you are pressing the Play (|>) button in "
                "your editor to start."
            )

        self.intervention_type = run_config["intervention_type"]
        self.no_graphics = run_config["no_graphics"]

        if side_channels is not None:
            self.stats_channel = side_channels[1]
            self.human_intervention_channel = side_channels[2]

        self.task_decay = task_decay

        # Create event Interpreter
        if self.intervention_type != "none":
            self.agent_interpreter = InterventionInterpreter(
                name=run_config["name"],
                model=run_config["model"],
                shot=run_config["shot"],
                intervention_type=run_config["intervention_type"],
            )

        import mlagents_envs
        from mlagents_envs.environment import UnityEnvironment

        # Try connecting to the Unity3D game instance. If a port is blocked
        port_ = None
        while True:
            # Sleep for random time to allow for concurrent startup of many
            # environments (num_env_runners >> 1). Otherwise, would lead to port
            # conflicts sometimes.
            if port_ is not None:
                time.sleep(random.randint(1, 10))
            port_ = port or (
                self._BASE_PORT_ENVIRONMENT if file_name else self._BASE_PORT_EDITOR
            )
            # cache the worker_id and
            # increase it for the next environment
            worker_id_ = Unity3DEnv._WORKER_ID if file_name else 0
            Unity3DEnv._WORKER_ID += 1
            try:
                self.unity_env = UnityEnvironment(
                    file_name=file_name,
                    worker_id=worker_id_,
                    base_port=port_,
                    seed=seed,
                    no_graphics=no_graphics,
                    timeout_wait=timeout_wait,
                    side_channels=side_channels,
                )
                print("Created UnityEnvironment for port {}".format(port_ + worker_id_))
            except mlagents_envs.exception.UnityWorkerInUseException:
                pass
            else:
                break

        # Reset entire env every this number of step calls.
        self.episode_horizon = episode_horizon
        # Keep track of how many times we have called `step` so far.
        self.episode_timesteps = 0

        self.total_time_steps = 0
        self.task_count = 0
        self.total_task_count = 0

        # self agent tracker: is agent following LLM instructions or receives actions from NN
        self.unity_env.reset()
        self.behavior_name = list(self.unity_env.behavior_specs)[0]
        agent_ids = list(self.unity_env.get_steps(self.behavior_name)[0].keys())
        self.all_agent_keys = [
            f"{self.behavior_name}_{agent_id}" for agent_id in agent_ids
        ]

        print(f"AGENT COUNT: {len(self.all_agent_keys)}")

        self.agent_policy_state = {}
        for agent_id in self.all_agent_keys:
            self.agent_policy_state[agent_id] = {
                "following": "NN",
                "task_decay": self.task_decay,
                "tasks": [],
            }

        # reset stats reader
        self.stats_channel.get_and_reset_stats()

        print("####################### ENVIRONMENT CREATED #######################")

    def update_occupied_agent_ids_list(self):
        occupied_agent_ids = [
            agent_id
            for agent_id, values in self.agent_policy_state.items()
            if values["following"] == "human"
            or values["following"] == "llm"
            or values["following"] == "auto"
        ]

        return occupied_agent_ids

    def asign_tasks_to_agents(self, tasks, occupied_agent_ids):
        for agent_id, task_and_intervention_type in tasks.items():
            if agent_id in self.all_agent_keys and agent_id not in occupied_agent_ids:
                task, intervention_type = task_and_intervention_type
                print(
                    f"┏ New Task-List for agent {YELLOW}{agent_id}{RESET} ---------------------------┑"
                )
                self.agent_policy_state[agent_id]["following"] = intervention_type
                # go to tasked location
                self.agent_policy_state[agent_id]["tasks"].append(task)
                print(f"agent {agent_id} tasked with go to location: {task}")
                if int(self.agent_interpreter.holding_water(agent_id)) == 0:
                    # get water
                    self.agent_policy_state[agent_id]["tasks"].append("water")
                    print(f"agent {agent_id} tasked with pick up water")
                    # and go back to tasked location
                    self.agent_policy_state[agent_id]["tasks"].append(task)
                    print(f"agent {agent_id} tasked with go back to location: {task}")

                print(
                    f"┗-------------------------------------------------------------------┙"
                )

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        """Performs one multi-agent step through the game.

        Args:
            action_dict: Multi-agent action dict with:
                keys=agent identifier consisting of
                [MLagents behavior name, e.g. "Goalie?team=1"] + "_" +
                [Agent index, a unique MLAgent-assigned index per single agent]

        Returns:
            tuple:
                - obs: Multi-agent observation dict.
                    Only those observations for which to get new actions are
                    returned.
                - rewards: Rewards dict matching `obs`.
                - dones: Done dict with only an __all__ multi-agent entry in
                    it. __all__=True, if episode is done for all agents.
                - infos: An (empty) info dict.
        """
        obs, _, _, _, _ = self._get_step_results()
        infos = {}
        occupied_agent_ids = self.update_occupied_agent_ids_list()

        if self.intervention_type != "none":
            self.agent_interpreter.get_all_observations(obs)

            # build task list if not NN only activated
            if len(occupied_agent_ids) < len(self.all_agent_keys):
                tasks = {}
                # HUMAN INTERVENTION
                if (
                    self.intervention_type == "human"
                    and self.human_intervention_channel.command != ""
                ):
                    print(f"HUMAN: {self.human_intervention_channel.command}")
                    response = self.agent_interpreter.human_intervention(
                        self.human_intervention_channel.get_and_reset_commands()
                    )
                    human_tasks = self.agent_interpreter.parse_task_interpretation(
                        response, self.intervention_type
                    )
                    tasks.update(human_tasks)
                # RULE-BASED OR LLM-BASED INTERVENTION
                elif len(self.agent_interpreter.all_agents_last_seen_fire) > 0:

                    self.agent_interpreter.observations_to_text(occupied_agent_ids)
                    if self.intervention_type == "auto":
                        prompt = self.agent_interpreter.rule_based_intervention()
                    if self.intervention_type == "llm":
                        prompt = self.agent_interpreter.llm_intervention(
                            observations=obs
                        )

                    print(
                        STRATEGY_PRINT.format(
                            STRATEGY=prompt,
                            RED=RED,
                            RESET=RESET,
                            TEMPERATURE=self.agent_interpreter.temperature,
                        )
                    )

                    response = self.agent_interpreter.run_LLM_interpreter(prompt)

                    print(
                        LLM_MEDIATOR_RESPONSE_PRINT.format(
                            GREEN=GREEN, RESET=RESET, RESPONSE=response
                        )
                    )
                    tasks = self.agent_interpreter.parse_task_interpretation(
                        response, self.intervention_type
                    )

                    if len(tasks) == 0:
                        # this is a safety measure to prevent the LLM from getting stuck
                        # with the same "faulty" strategy that is not parseable by the LLM-Mediator
                        self.agent_interpreter.temperature += 0.01

                if len(tasks) > 0:
                    self.task_count += len(tasks)
                    self.total_task_count += len(tasks)
                    self.asign_tasks_to_agents(tasks, occupied_agent_ids)
                    occupied_agent_ids = self.update_occupied_agent_ids_list()
                    self.agent_interpreter.temperature = 0.0

        # Set only the required actions (from the DecisionSteps) in Unity3D.
        # brain name: Agent
        actions = []
        message_to_agents = ""
        for agent_id in self.all_agent_keys:
            # agent_id = Agent?team=0_0
            # action_dict[agent_id] = (array([0.21004367], dtype=float32), array([0], dtype=int64))

            if agent_id not in action_dict:
                continue

            following = self.agent_policy_state[agent_id]["following"]

            if (
                following != "NN"
                and len(self.agent_policy_state[agent_id]["tasks"]) > 0
            ):
                # print(
                #     f"agent {agent_id} following: {following} - current task: {self.agent_policy_state[agent_id]['tasks'][0]} - got {self.agent_policy_state[agent_id]['task_decay']} steps left"
                # )

                # TODO:
                # send_string to Unity
                message_to_agents += f"{agent_id}:"

                # TASK: PICK WATER
                if self.agent_policy_state[agent_id]["tasks"][0] == "water":
                    self.task_pick_up_water(obs, agent_id, action_dict)
                # TASK: GO TO LOCATION
                else:
                    self.task_go_to_location(obs, agent_id, action_dict)

                infos[agent_id] = {following: action_dict[agent_id]}

                if len(self.agent_policy_state[agent_id]["tasks"]) == 0:
                    print(
                        f"agent {agent_id} finished all tasks after {self.task_decay - self.agent_policy_state[agent_id]['task_decay']} steps"
                    )
            else:
                # print(
                #     f"agent {agent_id} following NN - task decay: {self.agent_policy_state[agent_id]['task_decay']}"
                # )
                infos[agent_id] = {"NN": action_dict[agent_id]}

            # DECAY TASK
            if agent_id in occupied_agent_ids:
                self.agent_policy_state[agent_id]["task_decay"] -= 1
            actions.append(action_dict[agent_id])

        # if message_to_agents != "":
        #     intervention_sender = (
        #         "llm" if self.human_intervention_channel.command == "" else "human"
        #     )
        #     message_to_agents = f"{intervention_sender}#{message_to_agents[:-1]}"
        #     self.human_intervention_channel.send_string(message_to_agents)

        if len(actions) > 0:
            self.set_action(actions)

        # Do the step.
        self.unity_env.step()
        obs, rewards, terminateds, truncateds, _ = self._get_step_results()

        # check agents on LLM dutys decay or terminated and reset
        for agent_id, values in self.agent_policy_state.items():
            if terminateds["__all__"] or values["task_decay"] < 0:
                if terminateds["__all__"]:
                    print(f"agent {agent_id} terminated")
                if values["task_decay"] < 0:
                    print(
                        f"agent {agent_id} intervention cooldown below 0 - {len(values['tasks'])} task(s) left"
                    )

                self.reset_agent_policy_state(agent_id)

        self.episode_timesteps += 1
        self.total_time_steps += 1

        return obs, rewards, terminateds, truncateds, infos

    def reset(
        self, *, seed=None, options=None
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:

        print("####################### ENVIRONMENT RESET #######################")

        self.terminateds = set()
        self.truncateds = set()

        reset_stats_channel = self.episode_timesteps == 0

        """Resets the entire Unity3D scene (a single multi-agent episode)."""
        self.episode_timesteps = 0
        self.task_count = 0

        if reset_stats_channel:
            self.stats_channel.get_and_reset_stats()

        if self.intervention_type != "none":
            self.agent_interpreter.reset()

        for agent_id in self.all_agent_keys:
            self.agent_policy_state[agent_id] = {
                "following": "NN",
                "task_decay": self.task_decay,
                "tasks": [],
            }

        obs, _, _, dones, infos = self._get_step_results()

        if all(dones):
            self.unity_env.reset()

        return obs, infos

    def _get_step_results(self):
        """Collects those agents' obs/rewards that have to act in next `step`.

        Returns:
            Tuple:
                obs: Multi-agent observation dict.
                    Only those observations for which to get new actions are
                    returned.
                rewards: Rewards dict matching `obs`.
                dones: Done dict with only an __all__ multi-agent entry in it.
                    __all__=True, if episode is done for all agents.
                infos: An (empty) info dict.
        """
        obs = {}
        rewards = {}
        infos = {}

        # for behavior_name in self.unity_env.behavior_specs:
        decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)
        # ({'__all__': True, 'Agent?team=0_0': True, 'Agent?team=0_1': True, 'Agent?team=0_2': True},)
        # dones = dict({"__all__": False})
        for agent_id, idx in decision_steps.agent_id_to_index.items():
            key = self.behavior_name + "_{}".format(agent_id)
            # dones[key] = False
            os = tuple(o[idx] for o in decision_steps.obs)
            os = os[0] if len(os) == 1 else os
            obs[key] = os
            rewards[key] = decision_steps.reward[idx] + decision_steps.group_reward[idx]

        for agent_id, idx in terminal_steps.agent_id_to_index.items():
            key = self.behavior_name + "_{}".format(agent_id)
            # dones[key] = True
            # Only overwrite rewards (last reward in episode), b/c obs
            # here is the last obs (which doesn't matter anyways).
            # Unless key does not exist in obs.
            if key not in obs:
                os = tuple(o[idx] for o in terminal_steps.obs)
                os = os[0] if len(os) == 1 else os
                obs[key] = os
            rewards[key] = terminal_steps.reward[idx] + terminal_steps.group_reward[idx]

        if len(terminal_steps.interrupted) > 0:
            dones = dict(
                {"__all__": True},
                **{agent_id: True for agent_id in self.all_agent_keys},
            )
        else:
            dones = dict(
                {"__all__": False},
                **{agent_id: False for agent_id in self.all_agent_keys},
            )

        # Only use dones if all agents are done, then we should do a reset.
        return obs, rewards, dones, dones, infos

    @staticmethod
    def get_policy_configs_for_game(
        game_name: str, random: bool = False
    ) -> Tuple[dict, Callable[[AgentID], PolicyID]]:
        # The RLlib server must know about the Spaces that the Client will be
        # using inside Unity3D, up-front.
        obs_spaces = {
            "AerialWildfireSuppression": TupleSpace(
                [
                    Box(float(0), float(1), (42, 42, 3)),
                    Box(float("-inf"), float("inf"), (8,)),  # , dtype=np.float32),
                ]
            ),
        }
        action_spaces = {
            "AerialWildfireSuppression": TupleSpace(
                [
                    Box(float(-1), float(1), (1,), dtype=np.float32),
                    TupleSpace([Discrete(2)]),
                ]
            ),
        }

        if not random:
            policies = {
                game_name: PolicySpec(
                    policy_class=HIVEXPPOTorchPolicy,
                    observation_space=obs_spaces[game_name],
                    action_space=action_spaces[game_name],
                ),
            }
        else:
            policies = {
                "random_policy": PolicySpec(
                    policy_class=RandomPolicy,
                    observation_space=obs_spaces[game_name],
                    action_space=action_spaces[game_name],
                ),
            }

        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            return game_name if not random else "random_policy"

        return policies, policy_mapping_fn

    def task_pick_up_water(self, obs, agent_id, action_dict):
        # check if task is completed
        picked_water = (
            int(self.agent_interpreter.all_agents_holding_water[agent_id]) == 1
        )

        x = obs[agent_id][1][0] * 750
        y = obs[agent_id][1][1] * 750

        if picked_water == False:
            action_to_pick_up_water_from_closest_source = (
                self.get_action_to_pick_up_water_from_closest_source(obs, agent_id)
            )

            action_dict[agent_id] = (
                np.array(
                    [action_to_pick_up_water_from_closest_source],
                    dtype=np.float32,
                ),
                tuple([0]),
            )
        else:
            print(
                f"agent {agent_id} finished task picked up water at location: {x:.2f}, {y:.2f} after {self.task_decay - self.agent_policy_state[agent_id]['task_decay']} steps"
            )
            self.agent_policy_state[agent_id]["tasks"].pop(0)

    def task_go_to_location(self, obs, agent_id, action_dict):
        got_to_location = self.agent_policy_state[agent_id]["tasks"][0]
        go_to_x = got_to_location[0]
        go_to_y = got_to_location[1]
        target = [go_to_x, go_to_y]

        action_to_go_to_target = self.get_action_to_go_to_target(obs, agent_id, target)
        action_dict[agent_id] = (
            np.array([action_to_go_to_target], dtype=np.float32),
            tuple([0]),
        )

        distance_to_target = self.distance_to_target(obs, agent_id, target)
        # print(f"agent {agent_id} distance to target: {distance_to_target}")
        if distance_to_target < 25:
            print(
                f"agent {agent_id} finished task {self.agent_policy_state[agent_id]['tasks'][0]} after {self.task_decay - self.agent_policy_state[agent_id]['task_decay']} steps"
            )
            self.agent_policy_state[agent_id]["tasks"].pop(0)

    def set_action(self, actions):
        if isinstance(actions[0], tuple):
            action_tuple = ActionTuple(
                continuous=np.array([c[0] for c in actions]),
                discrete=np.array([c[1] for c in actions]),
            )
        else:
            if actions[0].dtype == np.float32:
                action_tuple = ActionTuple(continuous=np.array(actions))
            else:
                action_tuple = ActionTuple(discrete=np.array(actions))
        self.unity_env.set_actions(self.behavior_name, action_tuple)

    def reset_agent_policy_state(self, agent_id):
        self.agent_policy_state[agent_id]["following"] = "NN"
        self.agent_policy_state[agent_id]["task_decay"] = self.task_decay
        self.agent_policy_state[agent_id]["tasks"] = []

    def signed_angle_between_vectors(self, vector_a, vector_b):
        dot_product = np.dot(vector_a, vector_b)
        determinant = vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0]

        angle_rad = np.arctan2(determinant, dot_product)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def get_action_to_pick_up_water_from_closest_source(self, obs, agent_key):
        # float pos_x = Remap(this.transform.localPosition.x, -750f, 750f, -1f, 1f);
        # float pos_z = Remap(this.transform.localPosition.z, -750f, 750f, -1f, 1f);
        agent_location_x = obs[agent_key][1][0] * 750
        agent_location_y = obs[agent_key][1][1] * 750

        agent_location_x_abs = abs(agent_location_x)
        agent_location_y_abs = abs(agent_location_y)

        agent_dir_x = obs[agent_key][1][2]
        agent_dir_y = obs[agent_key][1][3]
        agent_dir = np.array([agent_dir_x, agent_dir_y])

        if agent_location_x_abs < agent_location_y_abs:
            # water closer in y
            if agent_location_y < 0:
                water_dir = np.array([0, -1])
            else:
                water_dir = np.array([0, 1])
        else:
            # water closer in x
            if agent_location_x < 0:
                water_dir = np.array([-1, 0])
            else:
                water_dir = np.array([1, 0])

        rotation = self.signed_angle_between_vectors(agent_dir, water_dir)

        # angle degree multiplier = 15
        action_multiplier = np.clip(rotation, -15, 15)
        action = remap(action_multiplier, -15, 15, -1, 1)

        return action

    def get_action_to_go_to_target(self, obs, agent_key, target):
        agent_location_x = obs[agent_key][1][0] * 750
        agent_location_y = obs[agent_key][1][1] * 750

        target_x = target[0] * 300
        target_y = target[1] * 300

        agent_to_target = np.array(
            [target_x - agent_location_x, target_y - agent_location_y]
        )
        agent_dir_x = obs[agent_key][1][2]
        agent_dir_y = obs[agent_key][1][3]
        agent_dir = np.array([agent_dir_x, agent_dir_y])

        rotation = self.signed_angle_between_vectors(agent_dir, agent_to_target)

        # angle degree multiplier = 15
        action_multiplier = np.clip(rotation, -15, 15)
        action = remap(action_multiplier, -15, 15, -1, 1)

        return action

    def distance_to_target(self, obs, agent_key, target):
        agent_location_x = obs[agent_key][1][0] * 750
        agent_location_y = obs[agent_key][1][1] * 750

        target_x = target[0] * 300
        target_y = target[1] * 300

        agent_to_target = np.array(
            [target_x - agent_location_x, target_y - agent_location_y]
        )

        length = np.linalg.norm(agent_to_target)

        return length


def remap(value, from1, to1, from2, to2):
    return (value - from1) / (to1 - from1) * (to2 - from2) + from2
