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

# Natural Language prompts with exact coordinates and flexible tasks
NL_STRATEGY_PROMPT = """### Instruction:
You are coordinating {AGENT_COUNT} aerial firefighting agents to extinguish wildfires and protect the village.

The environment is a 1500x1500 unit grid where:
- Coordinates range from -750 to 750 on both X and Y axes
- The village (high priority to protect) is in the bottom-right area (positive X, negative Y)
- Water sources are located at the edges of the map

Current agent states:
{AGENTS_DETAILED}

Fire locations:
{FIRES_DETAILED}

Water source locations are at the map edges (approximately at coordinates like (-700, 0), (700, 0), (0, -700), (0, 700)).

Create a strategy considering:
1. Agent proximity to fires and water sources
2. Whether agents are currently carrying water
3. Priority of protecting the village (bottom-right)
4. Agents can: go to specific coordinates, pick up water, or scout an area

### Response:
"""

NL_TASK_PROMPT = """You are assigning tasks to {AGENT_COUNT} aerial firefighting agents.

### Instruction:
Based on this strategy: '{STRATEGY}'

Assign ONE task per agent using EXACTLY one of these formats:
- <agent_name> goto <x> <y>  (go to exact coordinates, x and y are integers from -750 to 750)
- <agent_name> water  (go pick up water from nearest source)
- <agent_name> scout <x> <y>  (scout/patrol around coordinates)

Agent names: {AGENTS}

Respond with one task per line, using the exact agent names.

### Response:
"""
