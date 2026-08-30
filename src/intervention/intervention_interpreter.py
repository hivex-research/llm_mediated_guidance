import math
import numpy as np
import os
import re
import json
from pathlib import Path

from llm.registry import MODEL_REGISTRY
from intervention.prompt_templates import (
    FEW_SHOT_EXAMPLE_PROMPT,
    HUMAN_INTERVENTION_PROMPT,
    NL_INTERVENTION_PROMPT,
    NL_STRATEGY_PROMPT,
    NL_TASK_PROMPT,
    SAMPLE_PROMPT,
    STRATEGY_PROMPT,
)

DATA_PATH = Path(__file__).parent / "data"

HORIZONTAL = ["left", "center", "right"]
VERTICAL = ["bottom", "center", "top"]


class InterventionInterpreter(object):
    """
    OBSERVCATIONS:
    # k = 'Agent?team=0_0'
    # 0: pos_x,
    # 1: pos_y,
    # 2: dir_x,
    # 3: dir_y,
    # 4: water,
    # 5: last_burning_tree_x,
    # 6: last_burning_tree_y,
    # 7: last_burning_tree_bool
    # v = [-0.11725578, -0.8425961, 0., 0., 1., -1., -1., 0., 0.]
    """

    AGENT_IDS = None

    def __init__(
        self,
        name: str = "",
        llm_runner: str = None,
        stop_sequence: list = None,
        shot: str = None,
        intervention_type: str = "no",
        grid_size_x=1200,
        grid_size_y=1200,
        cell_count_x=3,
        cell_count_y=3,
    ):
        self.name = name

        model = MODEL_REGISTRY.get(llm_runner, None)
        if model:
            self.llm_runner = model()
            print(f"Using LLM model: {llm_runner}")
        else:
            self.llm_runner = None
            print(f"No LLM model found for: {llm_runner}")
        self.stop_sequence = ["###"] if stop_sequence is None else stop_sequence
        self.shot = "zero" if shot is None else shot
        self.intervention_type = intervention_type

        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.cell_count_x = cell_count_x
        self.cell_count_y = cell_count_y
        self.all_agents_fire_info = ""
        self.all_agents_location_info = ""
        self.all_agents_last_seen_fire = {}
        self.all_agents_location = {}
        self.all_agents_holding_water = {}
        self.all_agents_movement_dir = {}
        self.observation_scale_factor = 750

        self.temperature = 0.0

        self.output_dir = self.get_unique_output_dir(
            DATA_PATH / f"event_interpreter_data_{self.name}"
        )

    def human_intervention(self, text):
        prompt = HUMAN_INTERVENTION_PROMPT.format(
            LEN_AGENT_IDS=len(self.AGENT_IDS),
            TEXT=text,
        )

        return prompt

    def llm_intervention(self):
        """
        Natural Language intervention with exact coordinates and flexible task types.
        Agents can decide: go to location, pick up water, or scout.
        """
        agent_names = ", ".join([f"'{agent}'" for agent in self.AGENT_IDS])

        # Build detailed agent info with exact coordinates
        agents_detailed = self.get_detailed_agent_info()
        fires_detailed = self.get_detailed_fire_info()

        ### STRATEGY ###
        strategy_prompt = NL_STRATEGY_PROMPT.format(
            AGENT_COUNT=len(self.AGENT_IDS),
            AGENTS_DETAILED=agents_detailed,
            FIRES_DETAILED=fires_detailed,
        )

        strategy = self.llm_runner.run(strategy_prompt)

        ### TASK ASSIGNMENT ###
        task_prompt = NL_TASK_PROMPT.format(
            AGENT_COUNT=len(self.AGENT_IDS),
            STRATEGY=strategy,
            AGENTS=agent_names,
        )

        return task_prompt

    def get_detailed_agent_info(self):
        """Get detailed agent info with exact coordinates and water status."""
        info_lines = []
        for agent_id in self.AGENT_IDS:
            if agent_id in self.all_agents_location:
                x, y = self.all_agents_location[agent_id]
                has_water = int(self.all_agents_holding_water.get(agent_id, 0)) == 1
                water_status = "carrying water" if has_water else "no water"
                info_lines.append(
                    f"- {agent_id}: position ({x:.0f}, {y:.0f}), {water_status}"
                )
        return "\n".join(info_lines) if info_lines else "No agent data available"

    def get_detailed_fire_info(self):
        """Get detailed fire info with exact coordinates."""
        if not self.all_agents_last_seen_fire:
            return "No fires currently detected"

        # Deduplicate fires by rounding to avoid reporting same fire multiple times
        unique_fires = {}
        for agent_id, (fx, fy) in self.all_agents_last_seen_fire.items():
            # Round to nearest 50 units to deduplicate nearby fire reports
            key = (round(fx / 50) * 50, round(fy / 50) * 50)
            if key not in unique_fires:
                unique_fires[key] = (fx, fy)

        info_lines = []
        for idx, (fx, fy) in enumerate(unique_fires.values()):
            info_lines.append(f"- Fire {idx}: position ({fx:.0f}, {fy:.0f})")

        return "\n".join(info_lines)

    def rule_based_intervention(self):
        prompt = SAMPLE_PROMPT.format(
            ALL_AGENTS_LOCATION_INFO=self.all_agents_location_info,
            ALL_AGENTS_FIRE_INFO=self.all_agents_fire_info,
        )
        if self.shot == "few":
            prompt = FEW_SHOT_EXAMPLE_PROMPT + prompt

        return prompt

    def scripted_intervention(self, occupied_agent_ids=()):
        """Rule-Based Controller with the LLM removed.

        The Rule-Based directive is "instruct agent(s) to go to their closest
        fire", and the mediator's output for it is a 3x3 grid cell per agent.
        This computes that mapping directly -- nearest observed fire, then the
        grid cell containing it -- so the only difference from the `auto`
        pathway is whether a language model performs the mapping. It emits the
        same task format, so the environment side is untouched.
        """
        tasks = {}
        if not self.all_agents_last_seen_fire:
            return tasks

        fires = list(self.all_agents_last_seen_fire.values())
        for agent_id in self.AGENT_IDS or []:
            if agent_id in occupied_agent_ids:
                continue
            if agent_id not in self.all_agents_location:
                continue
            ax, ay = self.all_agents_location[agent_id]
            fx, fy = min(
                fires, key=lambda f: (f[0] - ax) ** 2 + (f[1] - ay) ** 2
            )
            index_x, index_y = self.get_grid_cell_from_position(fx, fy)
            # _parse_grid_tasks emits index-1, i.e. -1/0/+1; match it exactly.
            tasks[agent_id] = ((index_x - 1, index_y - 1), "auto")

        return tasks

    def parse_task_interpretation(self, response, intervention_type):
        if intervention_type == "human":
            agent_id_mapping = {
                "Agent?team=0_0": ["Agent 0", "Agent zero", "agent Zero", "Agent Zero"],
                "Agent?team=0_1": ["Agent 1", "Agent one", "agent One", "Agent One"],
                "Agent?team=0_2": ["Agent 2", "Agent two", "agent Two", "Agent Two"],
                "Agent?team=0_3": [
                    "Agent 3",
                    "Agent three",
                    "agent Three",
                    "Agent Three",
                ],
                "Agent?team=0_4": ["Agent 4", "Agent four", "agent Four", "Agent Four"],
                "Agent?team=0_5": ["Agent 5", "Agent five", "agent Five", "Agent Five"],
            }
            for key, values in agent_id_mapping.items():
                for value in values:
                    modified_values = [value]
                    if value.lower() not in modified_values:
                        modified_values.append(value.lower())
                    if value.capitalize() not in modified_values:
                        modified_values.append(value.capitalize())
                    for modified_value in modified_values:
                        if modified_value in response:
                            response = response.replace(modified_value, key)

        # Handle NL intervention with exact coordinates and task types
        if intervention_type == "llm":
            return self._parse_nl_tasks(response)

        # Original parsing for RB and human interventions
        return self._parse_grid_tasks(response, intervention_type)

    def _parse_nl_tasks(self, response):
        """
        Parse NL tasks with exact coordinates and task types.
        Formats:
        - <agent_name> goto <x> <y>
        - <agent_name> water
        - <agent_name> scout <x> <y>
        """
        response_lines = response.split("\n")
        agent_pattern = r"(Agent\?team=0_\d+)"
        tasks = {}

        for line in response_lines:
            if not line.strip():
                continue

            line = line.strip()

            # Find agent ID
            agent_match = re.search(agent_pattern, line)
            if not agent_match:
                continue
            agent_id = agent_match.group(1)

            # Determine task type and parse accordingly
            line_lower = line.lower()

            # Task: water (pick up water)
            if (
                " water" in line_lower
                and "goto" not in line_lower
                and "scout" not in line_lower
            ):
                tasks[agent_id] = ("water", "llm")
                continue

            # Task: goto <x> <y> or scout <x> <y>
            # Extract coordinates - look for two numbers
            numbers = re.findall(r"-?\d+\.?\d*", line)
            if len(numbers) >= 2:
                try:
                    x = float(numbers[-2])  # Second to last number
                    y = float(numbers[-1])  # Last number

                    # Clamp to valid range
                    x = max(-750, min(750, x))
                    y = max(-750, min(750, y))

                    # Normalize to -1 to 1 range for grid compatibility
                    norm_x = x / 750.0
                    norm_y = y / 750.0

                    if "scout" in line_lower:
                        tasks[agent_id] = (("scout", norm_x, norm_y), "llm")
                    else:
                        # Default to goto
                        tasks[agent_id] = (("goto", norm_x, norm_y), "llm")
                except (ValueError, IndexError):
                    continue

        return tasks

    def _parse_grid_tasks(self, response, intervention_type):
        """Original grid-based task parsing for RB and human interventions."""
        response_lines = response.split("\n")
        agent_pattern = r"(Agent\?team=0_\d+)"
        tasks = {}

        for index, agent_task in enumerate(response_lines):
            if agent_task == "":
                continue
            agent_task = agent_task.strip()
            words = agent_task.split(" ")
            agent_id = ""
            for word in words:
                agent_id = re.search(agent_pattern, word)
                if agent_id:
                    agent_id = word.replace("'", "")
                    words.remove(word)
                    break

            vertical_location = ""
            for word in words:
                clean_word = re.sub(r"[^a-zA-Z0-9\s]", "", word)
                clean_word = clean_word.lower()
                if clean_word in VERTICAL:
                    vertical_location = clean_word
                    words.remove(word)
                    break

            horizontal_location = ""
            for word in words:
                clean_word = re.sub(r"[^a-zA-Z0-9\s]", "", word)
                clean_word = clean_word.lower()
                if clean_word in HORIZONTAL:
                    horizontal_location = clean_word
                    words.remove(word)
                    break

            if (
                agent_id != ""
                and agent_id != None
                and vertical_location != ""
                and horizontal_location != ""
            ):
                tasks[agent_id] = (
                    (
                        HORIZONTAL.index(horizontal_location) - 1,
                        VERTICAL.index(vertical_location) - 1,
                    ),
                    intervention_type,
                )

        return tasks

    def get_unique_output_dir(self, base_dir):
        """
        Returns a unique directory name by adding a numerical suffix if the base_dir already exists.
        """
        if not os.path.exists(f"{base_dir}_0.json"):
            return f"{base_dir}_0.json"
        suffix = 1
        while os.path.exists(f"{base_dir}_{suffix}"):
            suffix += 1

        return f"{base_dir}_{suffix}.json"

    def run_LLM_interpreter(self, prompt):

        completion = self.llm_runner.run(prompt)

        data = {
            "all_agents_location_info": self.all_agents_location_info,
            "all_agents_fire_info": self.all_agents_fire_info,
            "prompt": prompt,
            "completion": completion,
        }

        try:
            os.makedirs(DATA_PATH, exist_ok=True)
            with open(
                self.output_dir,
                "a",
            ) as file:
                json.dump(data, file, indent=4)
        except Exception as e:
            print(f"An error occurred: {e}")

        return completion

    def get_all_observations(self, obs):
        if self.AGENT_IDS is None:
            self.AGENT_IDS = [f"Agent?team=0_{x}" for x in range(len(obs))]

        self.get_all_agents_last_seen_fire_location(obs)
        self.get_all_agents_location(obs)
        self.get_all_agents_movement_dir(obs)
        self.get_all_agents_holding_water(obs)

    def observations_to_text(self, agent_ids_with_task=[]):
        self.get_all_agent_observed_fire()
        self.get_all_agent_location(agent_ids_with_task)

    def get_all_agents_last_seen_fire_location(self, obs):
        self.all_agents_last_seen_fire = {}
        for k, v in obs.items():
            vector_obs = v[1]

            if vector_obs[7] != 0:
                self.all_agents_last_seen_fire[k] = (
                    vector_obs[5] * self.observation_scale_factor,
                    vector_obs[6] * self.observation_scale_factor,
                )

    def get_all_agents_location(self, obs):
        for k, v in obs.items():
            vector_obs = v[1]
            self.all_agents_location[k] = (
                vector_obs[0] * self.observation_scale_factor,
                vector_obs[1] * self.observation_scale_factor,
            )

    def get_all_agents_movement_dir(self, obs):
        for k, v in obs.items():
            vector_obs = v[1]
            self.all_agents_movement_dir[k] = (vector_obs[2], vector_obs[3])

    def get_all_agents_holding_water(self, obs):
        for k, v in obs.items():
            vector_obs = v[1]
            self.all_agents_holding_water[k] = vector_obs[4]

    def get_observed_fire(self, x, y):
        fire_location_indexes = self.get_grid_cell_from_position(x, y)
        return f"{VERTICAL[fire_location_indexes[1]]} {HORIZONTAL[fire_location_indexes[0]]}"

    def get_all_agent_observed_fire(self):
        info_list = [
            self.get_observed_fire(x=observed_fire[0], y=observed_fire[1])
            for _, observed_fire in self.all_agents_last_seen_fire.items()
        ]

        info_list = list(set(info_list))
        info_list = [
            f"Fire {index} is in the {info}" for index, info in enumerate(info_list)
        ]
        fire_count = len(info_list)

        info = " and ".join(info_list)

        if info != "":
            info = f"{fire_count} fire(s): {info}."
        else:
            info = ""

        self.all_agents_fire_info = info

    def get_all_agent_location(self, agent_ids_with_task=[]):
        agent_location_indexes = {
            k: self.get_grid_cell_from_position(agent_position[0], agent_position[1])
            for k, agent_position in self.all_agents_location.items()
            if k not in agent_ids_with_task
        }

        location_info = ", ".join(
            [
                f"Agent '{key}' is in the {VERTICAL[index[1]]} {HORIZONTAL[index[0]]}"
                for key, index in agent_location_indexes.items()
            ]
        )
        location_info += "."

        self.all_agents_location_info = location_info

    def get_grid_cell_from_position(self, x, y):
        half_extend_x = self.grid_size_x / 2
        half_extend_y = self.grid_size_y / 2

        x = np.clip(x, -(half_extend_x - 1), half_extend_x - 1)
        y = np.clip(y, -(half_extend_y - 1), half_extend_y - 1)

        index_x = math.floor(
            remap(x, -half_extend_x, half_extend_x, 0.0, self.cell_count_x)
        )
        index_y = math.floor(
            remap(y, -half_extend_y, half_extend_y, 0.0, self.cell_count_y)
        )

        # return cell indexes
        # 0,2|1,2|2,2
        # 0,1|1,1|2,1
        # 0,0|1,0|2,0
        return (index_x, index_y)

    def reset(self):
        print(f"Intervention Interpreter Reset")

        self.all_agents_fire_info = ""
        self.all_agents_location_info = ""
        self.all_agents_last_seen_fire = {}
        self.all_agents_location = {}
        self.all_agents_holding_water = {}
        self.all_agents_movement_dir = {}

    def holding_water(self, agent_id: str):
        return self.all_agents_holding_water[agent_id]


def remap(value, from1, to1, from2, to2):
    return (value - from1) / (to1 - from1) * (to2 - from2) + from2
