STRATEGY_PROMPT = """### Instruction:
{AGENT_NAMES} each control an aeroplane to extinguish fire and protect the village, which is of high importance. Your task is it to come up with a strategy for the agents movements on a 3 by 3 grid:
| top left | top center | top right |
| center left | center center | center right |
| bottom left | bottom center | bottom right |
The following are the positions of all agents and fire locations:
{AGENTS}
{FIRES}
The village is in the bottomn right.
Please use locations combinations HORIZONTAL VERTICAL and correct agent names in your description.
### Response:
"""

NL_INTERVENTION_PROMPT = """You are controlling {AGENT_COUNT} agents in an autonomous aerial wildfire suppression scenario
### Instruction:
Locations:
HORIZONTAL locations: ['left', 'right', 'center']
VERTICAL locations: ['bottom', 'top', 'center']
Parse this user input: '{STRATEGY}'. Use this task template: '<agent name> go to <VERTICAL location> <HORIZONTAL location>'. For example:
Agent 'Agent?team=0_0' go to {EXAMPLE}
Use one line per agent and be precise with the template. Use the correct agent names: {AGENTS}
### Response:
"""

HUMAN_INTERVENTION_PROMPT = """You are controlling {LEN_AGENT_IDS} agents in an autonomous aerial wildfire suppression scenario
### Instruction:        
Locations:
HORIZONTAL locations: ["left", "right", "center"]
VERTICAL locations: ["bottom", "top", "center"]
Summarize this user input: "{TEXT}". Use this task template: <agent name> go to <VERTICAL location> <HORIZONTAL location>
Use one line per agent and be precise with the template.        
### Response:
"""

FEW_SHOT_EXAMPLE_PROMPT = """### Instruction:
HORIZONTAL locations: ['left', 'right', 'center']
VERTICAL locations: ['bottom', 'top', 'center']
Instruct agent(s) to go to their closest fire.
Reply like this: <agent name> go to <VERTICAL location> <HORIZONTAL location>
### Input:
Agent 'Agent?team=0_0' is in the bottom center, Agent 'Agent?team=0_1' is in the center right, Agent 'Agent?team=0_2' is in the bottom center. 1 fire(s): Fire 0 is in the center center.
Where should agents go?
Agent 'Agent?team=0_0' go to center center
Agent 'Agent?team=0_1' go to center center
Agent 'Agent?team=0_2' go to center center
"""

SAMPLE_PROMPT = """### Instruction:
HORIZONTAL locations: ['left', 'right', 'center']
VERTICAL locations: ['bottom', 'top', 'center']
Instruct agent(s) to go to their closest fire.
Reply like this: <agent name> go to <VERTICAL location> <HORIZONTAL location>
### Input:
{ALL_AGENTS_LOCATION_INFO} {ALL_AGENTS_FIRE_INFO}
Where should agents go?
"""
