import math
import numpy as np
import os
import re
import json
from pathlib import Path
from typing import Sequence
from dotenv import load_dotenv
from aleph_alpha_client import Prompt, SemanticEmbeddingRequest, SemanticRepresentation

DATA_PATH = Path(__file__).parent / "data"
# Load the .env file
load_dotenv()

# # OPENAI
# from openai import OpenAI

# # Get the API key from the environment variable
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# # Set up the API key
# open_ai_client = OpenAI(api_key=OPENAI_API_KEY)

# ALEPH ALPHA
from aleph_alpha_client import Client, Prompt, CompletionRequest

API_KEY = os.getenv("AA_TOKEN")
aleph_alpha_model = Client(token=API_KEY)

HORIZONTAL = ["left", "center", "right"]
VERTICAL = ["bottom", "center", "top"]

AGENT_IDS = [
    "Agent?team=0_0",
    "Agent?team=0_1",
    "Agent?team=0_2",
]

STRATEGY_PROMPT = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Instruction:
Agent 'Agent?team=0_0', Agent 'Agent?team=0_1' and Agent 'Agent?team=0_2' each control an aeroplane to extinguish fire and protect the village, which is of high importance. Your task is it to come up with a strategy for the agents movements on a 3 by 3 grid:
| top left | top center | top right |
| center left | center center | center right |
| bottom left | bottom center | bottom right |
The following are the positions of all agents and fire locations:
{AGENTS}
{FIRES}
The village is in the bottomn right.
Please use locations combinations HORIZONTAL VERTICAL and correct agent names in your description.
Response:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

NL_INTERVENTION_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are controlling 3 agents in an autonomous aerial wildfire suppression scenario<|eot_id|><|start_header_id|>user<|end_header_id|>
Instruction:
Locations:
HORIZONTAL locations: ['left', 'right', 'center']
VERTICAL locations: ['bottom', 'top', 'center']
Parse this user input: '{STRATEGY}'. Use this task template: '<agent name> go to <VERTICAL location> <HORIZONTAL location>'. For example:
Agent 'Agent?team=0_0' go to {EXAMPLE}
Use one line per agent and be precise with the template. Use the correct agent names: 'Agent?team=0_0', 'Agent?team=0_1', 'Agent?team=0_2'
Response:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

# -1, 1 | 0, 1 | 1, 1
# -1, 0 | 0, 0 | 1, 0
# -1, -1| 0, -1| 1, -1


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

    def __init__(
        self,
        name: str = "",
        model: str = None,
        stop_sequence: list = None,
        shot: str = None,
        intervention_type: str = "no",
        grid_size_x=1200,
        grid_size_y=1200,
        cell_count_x=3,
        cell_count_y=3,
    ):
        self.name = name
        # self.model = "luminous-extended-control" if model is None else model
        self.model = (
            model  # "Pharia-1-LLM-7B-control-aligned"  # "llama-3.1-8b-instruct"
        )
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

        # self.asymmetric_embeddings = [
        #     self.embed(text, SemanticRepresentation.Document) for text in AGENT_IDS
        # ]

    def aleph_alpha_api_worker(self, prompt, maximum_tokens=60):
        # AlephAlpha
        params = {
            "prompt": Prompt.from_text(prompt),
            "maximum_tokens": maximum_tokens,
            "temperature": self.temperature,
        }

        request = CompletionRequest(**params)
        response = aleph_alpha_model.complete(request, model=self.model)

        return response.completions[0].completion

    def human_intervention(self, text):

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are controlling 3 agents in an autonomous aerial wildfire suppression scenario<|eot_id|><|start_header_id|>user<|end_header_id|>
        ### Instruction:        
        Locations:
        HORIZONTAL locations: ["left", "right", "center"]
        VERTICAL locations: ["bottom", "top", "center"]
        Summarize this user input: "{text}". Use this task template: <agent name> go to <VERTICAL location> <HORIZONTAL location>
        Use one line per agent and be precise with the template.        
        ### Response:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        return prompt

    def llm_intervention(self, observations):

        ### STRATEGY ###

        intervention_prompt = STRATEGY_PROMPT.format(
            AGENTS=self.all_agents_location_info, FIRES=self.all_agents_fire_info
        )

        strategy = self.aleph_alpha_api_worker(intervention_prompt, 196)

        ### INTERVENTION ###

        interpreter_prompt = NL_INTERVENTION_PROMPT.format(
            STRATEGY=strategy,
            EXAMPLE="bottom right" if "Pharia" in self.model else "y x",
        )

        return interpreter_prompt

    def rule_based_intervention(self):
        few_shot = """<|start_header_id|>user<|end_header_id|>\nInstruction:\nHORIZONTAL locations: ['left', 'right', 'center']\nVERTICAL locations: ['bottom', 'top', 'center']\nInstruct agent(s) to go to their closest fire.\nReply like this: <agent name> go to <VERTICAL location> <HORIZONTAL location>\nInput:\nAgent 'Agent?team=0_0' is in the bottom center, Agent 'Agent?team=0_1' is in the center right, Agent 'Agent?team=0_2' is in the bottom center. 1 fire(s): Fire 0 is in the center center.\nWhere should agents go?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\nAgent 'Agent?team=0_0' go to center center\nAgent 'Agent?team=0_1' go to center center\nAgent 'Agent?team=0_2' go to center center<|eot_id|>"""

        sample = f"<|start_header_id|>user<|end_header_id|>\nInstruction:\nHORIZONTAL locations: ['left', 'right', 'center']\nVERTICAL locations: ['bottom', 'top', 'center']\nInstruct agent(s) to go to their closest fire.\nReply like this: <agent name> go to <VERTICAL location> <HORIZONTAL location>\nInput:\n{self.all_agents_location_info} {self.all_agents_fire_info}\nWhere should agents go?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

        if self.shot == "few":
            prompt = "<|begin_of_text|>" + few_shot + sample
        else:
            prompt = "<|begin_of_text|>" + sample

        return prompt

    def parse_task_interpretation(self, response, intervention_type):
        if intervention_type == "human":
            agent_id_mapping = {
                "Agent?team=0_0": ["Agent 0", "Agent zero", "agent Zero", "Agent Zero"],
                "Agent?team=0_1": ["Agent 1", "Agent one", "agent One", "Agent One"],
                "Agent?team=0_2": ["Agent 2", "Agent two", "agent Two", "Agent Two"],
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

        response = response.split("\n")

        agent_pattern = r"(Agent\?team=0_\d+)"

        tasks = {}

        for index, agent_task in enumerate(response):
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

            # if agent_id could not be found using pattern use cosine similarity
            # if agent_id not in AGENT_IDS:
            #     asymmetric_query = self.embed(agent_task, SemanticRepresentation.Query)
            #     max_cosine_similarity_score = 0
            #     max_cosine_similarity_agent_id = ""
            #     for i, id in enumerate(AGENT_IDS):
            #         cosine_similarity_score = self.cosine_similarity(
            #             asymmetric_query, self.asymmetric_embeddings[i]
            #         )
            #         if max_cosine_similarity_score < cosine_similarity_score:
            #             max_cosine_similarity_score = cosine_similarity_score
            #             max_cosine_similarity_agent_id = id

            #     if max_cosine_similarity_score > 0.4:
            #         agent_id = max_cosine_similarity_agent_id
            #         print(
            #             f"after using cosine similarity, found agent_id: {agent_id} in command: {agent_task}"
            #         )

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

        # example task list
        # { 'Agent?team=0_0': ('bottom', 'left'), 'Agent?team=0_1': ('center', 'center') }
        # HORIZONTAL = ["left", "center", "right"]
        # VERTICAL = ["bottom", "center", "top"]

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

        completion = self.aleph_alpha_api_worker(prompt)

        data = {
            "all_agents_location_info": self.all_agents_location_info,
            "all_agents_fire_info": self.all_agents_fire_info,
            "prompt": prompt,
            "completion": completion,
        }

        try:
            # Ensure the directory exists
            os.makedirs(DATA_PATH, exist_ok=True)
            # Writing JSON data
            with open(
                self.output_dir,
                "a",
            ) as file:
                json.dump(data, file, indent=4)
            # print(
            #     f"File 'event_interpreter_data_{self.name}.json' has been created in the directory: {DATA_PATH}"
            # )
        except Exception as e:
            print(f"An error occurred: {e}")

        return completion

    def embed(self, text: str, representation: SemanticRepresentation):
        request = SemanticEmbeddingRequest(
            prompt=Prompt.from_text(text), representation=representation
        )
        result = aleph_alpha_model.semantic_embed(request, model="luminous-base")
        return result.embedding

    def cosine_similarity(self, v1: Sequence[float], v2: Sequence[float]) -> float:
        "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(v1)):
            x = v1[i]
            y = v2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
        return sumxy / math.sqrt(sumxx * sumyy)

    def get_all_observations(self, obs):
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
        # fire grid cell indexes
        fire_location_indexes = self.get_grid_cell_from_position(x, y)
        # "There is fire in the bottom right"
        return f"{VERTICAL[fire_location_indexes[1]]} {HORIZONTAL[fire_location_indexes[0]]}"

    def get_all_agent_observed_fire(self):
        # fire grid cell indexes
        info_list = [
            self.get_observed_fire(x=observed_fire[0], y=observed_fire[1])
            for _, observed_fire in self.all_agents_last_seen_fire.items()
        ]
        # dedup
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
