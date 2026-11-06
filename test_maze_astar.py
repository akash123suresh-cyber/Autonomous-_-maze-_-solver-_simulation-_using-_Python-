import ast
import heapq
from pathlib import Path

source = Path.home() / "autonomous_maze_robot/visualization/animated_search.py"

tree = ast.parse(source.read_text())

maze_node = next(
    node for node in tree.body
    if isinstance(node, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "maze" for t in node.targets)
)

maze = ast.literal_eval(maze_node.value.args[0])

start = (1, 1)
goal = (19, 29)

rows = len(maze)
cols = len(maze[0])

directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def valid(pos):
    r, c = pos
    return (
        0 <= r < rows
        and 0 <= c < cols
        and maze[r][c] == 0
    )


def heuristic(pos):
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


def astar():
    open_list = [(heuristic(start), 0, start)]
    g_cost = {start: 0}
    parent = {}
    closed = set()
    explored = []

    while open_list:
        _, cost, current = heapq.heappop(open_list)

        if current in closed:
            continue

        closed.add(current)
        explored.append(current)

        if current == goal:
            break

        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)

            if not valid(neighbor):
                continue

            new_cost = g_cost[current] + 1

            if new_cost < g_cost.get(neighbor, float("inf")):
                g_cost[neighbor] = new_cost
                parent[neighbor] = current

                heapq.heappush(
                    open_list,
                    (new_cost + heuristic(neighbor), new_cost, neighbor)
                )

    if goal not in closed:
        return [], explored

    path = [goal]

    while path[-1] != start:
        path.append(parent[path[-1]])

    path.reverse()
    return path, explored


print("Start traversable:", valid(start))
print("Goal traversable:", valid(goal))

path, explored = astar()

print("Grid dimensions:", rows, "x", cols)
print("Cells explored:", len(explored))

if path:
    print("A* found a path!")
    print("Path length in moves:", len(path) - 1)
    print("Start:", path[0])
    print("Goal:", path[-1])
    print("Path:", path)
else:
    print("No path found. Check the maze and start/goal cells.")
