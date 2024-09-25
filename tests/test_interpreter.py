import pytest
from intervention.intervention_interpreter import (
    InterventionInterpreter,
)


def test_get_observed_fire():
    agent_interpreter = InterventionInterpreter()
    info = agent_interpreter.get_observed_fire(x=-450, y=-550)
    # There is fire in the bottom left
    print(info)
    info = agent_interpreter.get_observed_fire(x=32, y=17)
    # There is fire in the middle middle
    print(info)
    info = agent_interpreter.get_observed_fire(x=450, y=550)
    # There is fire in the top right
    print(info)


def test_get_all_agent_observed_fire():
    agent_interpreter = InterventionInterpreter()
    all_agent_observed_fire_location = [(-450, 550), (32, 17), (450, 550)]
    info = agent_interpreter.get_all_agent_location(
        all_agent_observed_fire_location=all_agent_observed_fire_location
    )
    print(info)


def test_interpret_observed_agent_location():
    agent_interpreter = InterventionInterpreter()
    agent_locations = [(-450, 32), (360, -450), (550, -32)]
    position_info = agent_interpreter.get_all_agent_location(agent_locations)
    # Agent 0 is in the middle left, Agent 1 is in the bottom right, Agent 2 is in the middle right
    print(position_info)


test_cases = [
    # {
    #     "fire_locations": {
    #         "Agent?team=0_0": (-450, -550),
    #         "Agent?team=0_1": (0, 0),
    #         "Agent?team=0_2": (0, 0),
    #     },
    #     "agent_locations": {
    #         "Agent?team=0_0": (-450, 32),
    #         "Agent?team=0_1": (360, -450),
    #         "Agent?team=0_2": (550, -32),
    #     },
    #     "solution": {"Agent?team=0_0": (-1, -1), "Agent?team=0_1": (0, 0)},
    # },
    # {
    #     "fire_locations": {
    #         "Agent?team=0_0": (-450, -550),
    #         "Agent?team=0_1": (450, 0),
    #         "Agent?team=0_2": (0, 0),
    #     },
    #     "agent_locations": {
    #         "Agent?team=0_0": (-450, 32),
    #         "Agent?team=0_1": (360, -450),
    #         "Agent?team=0_2": (550, -32),
    #     },
    #     "solutions": [
    #         {"Agent?team=0_0": (1, 0), "Agent?team=0_1": (0, 0)},
    #         {"Agent?team=0_0": (-1, -1), "Agent?team=0_1": (0, 0)},
    #     ],
    # },
    {
        "fire_locations": {
            "Agent?team=0_0": (-450, -550),
            "Agent?team=0_1": (0, 0),
            "Agent?team=0_2": (0, 0),
        },
        "agent_locations": {
            "Agent?team=0_0": (-450, 32),
            "Agent?team=0_1": (360, -450),
            "Agent?team=0_2": (550, -32),
        },
        "solutions": [
            {"Agent?team=0_0": (1, 0), "Agent?team=0_1": (0, 0)},
            {"Agent?team=0_0": (-1, -1), "Agent?team=0_1": (0, 0)},
        ],
    },
]


@pytest.mark.parameterize("test_cases", test_cases)
def test_LLM_info_to_direct_task_interpreter(test_cases):
    agent_interpreter = InterventionInterpreter()

    # FIRE LOCATIONS
    all_agents_last_seen_fire_location = test_cases["fire_locations"]
    agent_interpreter.all_agents_last_seen_fire = all_agents_last_seen_fire_location
    agent_interpreter.get_all_agent_observed_fire()

    # AGENT LOCATIONS
    all_agents_location = test_cases["agent_locations"]
    agent_interpreter.all_agents_location = all_agents_location
    keys_to_exclude = []  #  ["Agent?team=0_2"]  # []
    agent_interpreter.get_all_agent_location(keys_to_exclude)

    response = agent_interpreter.rule_based_intervention_to_llm_task_interpretation()
    print(response)
    tasks = agent_interpreter.parse_task_interpretation(response)
    print(tasks)


def test_human_intervention_to_llm_task_interpretation(message):
    agent_interpreter = InterventionInterpreter()

    response = agent_interpreter.human_intervention_to_llm_task_interpretation(message)

    print(response)
