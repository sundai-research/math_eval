from pathlib import Path
import json
from typing import TypedDict
import asyncio

"""
{"type": "function", "function": {"name": "circle_distance_minimizer", "description": "Finds the minimum sum of distances from a point on a circle to two fixed points using numerical optimization", "parameters": {"type": "object", "properties": {"circle_center": {"type": "array", "description": "Coordinates [x, y] of the circle's center"}, "radius": {"type": "float", "description": "Radius of the circle"}, "point1": {"type": "array", "description": "Coordinates [x, y] of the first fixed point"}, "point2": {"type": "array", "description": "Coordinates [x, y] of the second fixed point"}}, "required": ["circle_center", "radius", "point1", "point2"]}}, "python_implementation": "async def circle_distance_minimizer(circle_center, radius, point1, point2):\n    \"\"\"Computes minimum sum of distances from circle to two points using numerical optimization\"\"\"\n    import numpy as np\n    from scipy.optimize import minimize_scalar\n    \n    def distance_sum(theta):\n        x = circle_center[0] + radius * np.cos(theta)\n        y = circle_center[1] + radius * np.sin(theta)\n        d1 = np.sqrt((x - point1[0])**2 + (y - point1[1])**2)\n        d2 = np.sqrt((x - point2[0])**2 + (y - point2[1])**2)\n        return d1 + d2\n    \n    result = minimize_scalar(distance_sum, bounds=(0, 2*np.pi), method='bounded')\n    return {\n        'minimum_value': result.fun,\n        'theta': result.x,\n        'steps': ['Parametrized circle', 'Computed distance sum', 'Optimized using scalar minimization']\n    }", "required_imports": ["numpy", "scipy.optimize"]}
"""


# - add a tool to check if the answer is correct
# - add a tool to check if the answer is correct
class ToolSchema(TypedDict):
    name: str


class ToolDefinition(TypedDict):
    type: str
    function: ToolSchema
    required_imports: list[str]
    python_implementation: str


class ToolManager:
    """
    The ToolManager is responsible for:
    - Writing new tools to the tool file
    - Executing any tools that exist in the file
    - Reading and providing information about the tools

    Really important to keep this simple
    """

    def __init__(self, tools_file: str):
        self.tools_file = Path(tools_file)

        # create the file if it doesn't exist yet
        if not self.tools_file.exists():
            self.tools_file.write_text("", "utf-8")

    def read_tools(self) -> list[ToolDefinition]:
        existing_tools = [
            ToolDefinition(json.loads(t))
            for t in self.tools_file.read_text().splitlines()
        ]
        return existing_tools

    def get_tool(self, tool_name: str) -> ToolDefinition | None:
        """
        Return the tool object definition for a given tool else None
        """
        tools = self.read_tools()
        matching_tools = [
            tool for tool in tools if tool["function"]["name"] == tool_name
        ]
        if not matching_tools:
            return None
        return matching_tools[0]

    def tool_exists(self, tool_name: str) -> bool:
        return self.get_tool(tool_name) is not None

    def execute_tool(self, name: str, **params) -> any:
        """
        Given a tool name, execute it with the given parameter set and return result
        """
        # get the tool
        tool = self.get_tool(name)
        if not tool:
            raise ValueError("this tool does not exist in the database")

        # if the tool exists, load it in and try calling it
        exec(tool["python_implementation"], globals())
        func = globals()[name]
        print(f"{params=}")
        return func(**params)


if __name__ == "__main__":
    mgr = ToolManager("/mnt/7TB-a/osilkin/math_eval/math_tools_20250629_200159.jsonl")

    tool_names = [
        "TrigExpressionSimplifier",
        "check_skew_perpendicular_lines",
        "NonIntersectingPathsCalculator",
        "circle_distance_minimizer",
    ]
    for tn in tool_names:
        assert mgr.tool_exists(tn)
        tool = mgr.get_tool(tn)

    params = {
        "circle_center": [0, 0],
        "radius": 1,
        "point1": [0.5, 0.1],
        "point2": [0.1, 0.5],
    }
    result = mgr.execute_tool("circle_distance_minimizer", **params)
    print(result)
