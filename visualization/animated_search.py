import sys
import numpy as np
import heapq
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# MAZE
# 1 = Wall
# 0 = Free space
# ============================================================

maze = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,1,0,1,0,1,1,1,1,0,1,0,1,1,1,1,0,1,0,1,1,1,1,1,1,1,0,1],
    [1,0,1,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1],
    [1,0,1,0,1,1,1,0,1,0,1,1,1,1,0,1,0,1,1,1,1,0,1,0,1,1,1,0,1,0,1],
    [1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,1,0,1],
    [1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,1,1,1,0,1,0,1,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,1,0,1,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,1,1,1,1,1,0,1,0,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
    [1,0,1,1,1,1,1,1,1,1,0,1,0,1,0,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1],
    [1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,0,1],
    [1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,1],
    [1,0,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,1,1,1,1,1,1,1,0,1],
    [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
])

start = (1, 1)
goal = (19, 29)

directions = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1)
]


# ============================================================
# CHECK VALID POSITION
# ============================================================

def is_valid(position):

    row, col = position

    if row < 0 or row >= maze.shape[0]:
        return False

    if col < 0 or col >= maze.shape[1]:
        return False

    return maze[row, col] == 0


# ============================================================
# DFS
# ============================================================

def dfs():

    stack = [start]

    visited = {start}

    parent = {}

    explored = []

    while stack:

        current = stack.pop()

        explored.append(current)

        if current == goal:
            break

        for direction in directions:

            neighbor = (
                current[0] + direction[0],
                current[1] + direction[1]
            )

            if is_valid(neighbor) and neighbor not in visited:

                visited.add(neighbor)

                parent[neighbor] = current

                stack.append(neighbor)

    return reconstruct_path(parent), explored


# ============================================================
# BFS
# ============================================================

def bfs():

    queue = deque([start])

    visited = {start}

    parent = {}

    explored = []

    while queue:

        current = queue.popleft()

        explored.append(current)

        if current == goal:
            break

        for direction in directions:

            neighbor = (
                current[0] + direction[0],
                current[1] + direction[1]
            )

            if is_valid(neighbor) and neighbor not in visited:

                visited.add(neighbor)

                parent[neighbor] = current

                queue.append(neighbor)

    return reconstruct_path(parent), explored


# ============================================================
# A*
# ============================================================

def heuristic(position):

    return (
        abs(position[0] - goal[0])
        + abs(position[1] - goal[1])
    )


def astar():

    open_list = []

    heapq.heappush(
        open_list,
        (heuristic(start), start)
    )

    g_cost = {start: 0}

    parent = {}

    explored = []

    closed = set()

    while open_list:

        f_cost, current = heapq.heappop(open_list)

        if current in closed:
            continue

        closed.add(current)

        explored.append(current)

        if current == goal:
            break

        for direction in directions:

            neighbor = (
                current[0] + direction[0],
                current[1] + direction[1]
            )

            if not is_valid(neighbor):
                continue

            new_g = g_cost[current] + 1

            if neighbor not in g_cost or new_g < g_cost[neighbor]:

                g_cost[neighbor] = new_g

                parent[neighbor] = current

                f = new_g + heuristic(neighbor)

                heapq.heappush(
                    open_list,
                    (f, neighbor)
                )

    return reconstruct_path(parent), explored


# ============================================================
# RECONSTRUCT PATH
# ============================================================

def reconstruct_path(parent):

    path = []

    if goal not in parent and goal != start:
        return path

    current = goal

    while current != start:

        path.append(current)

        current = parent[current]

    path.append(start)

    path.reverse()

    return path


# ============================================================
# RUN ALL ALGORITHMS
# ============================================================

algorithms = {
    "DFS": dfs,
    "BFS": bfs,
    "A*": astar
}

results = {}

for name, algorithm in algorithms.items():

    path, explored = algorithm()

    results[name] = {
        "path": path,
        "explored": explored
    }


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 60)
print("             ANIMATED MAZE SOLVER")
print("=" * 60)

for name, result in results.items():

    print(
        f"{name}: "
        f"Explored = {len(result['explored'])}, "
        f"Path Length = {len(result['path'])}"
    )

print("=" * 60)


# ============================================================
# ANIMATION
# ============================================================

# Select one algorithm from the terminal
available_algorithms = ["DFS", "BFS", "A*"]

if len(sys.argv) > 1:
    selected = sys.argv[1].upper()

    if selected == "ASTAR":
        selected = "A*"

    if selected not in available_algorithms:
        raise SystemExit(
            "Invalid algorithm. Choose: DFS, BFS, or A*"
        )

    algorithm_names = [selected]

else:
    algorithm_names = available_algorithms

current_algorithm = 0
current_step = 0

fig, ax = plt.subplots(figsize=(9, 7))


def draw_frame():

    global current_algorithm
    global current_step

    name = algorithm_names[current_algorithm]

    result = results[name]

    explored = result["explored"]

    path = result["path"]

    ax.clear()

    # Draw maze
    ax.imshow(
        maze,
        cmap="gray_r"
    )

    # --------------------------------------------------------
    # Draw explored cells
    # --------------------------------------------------------

    visible_explored = explored[:current_step]

    for row, col in visible_explored:

        if (row, col) != start and (row, col) != goal:

            ax.scatter(
                col,
                row,
                c="yellow",
                s=25
            )

    # --------------------------------------------------------
    # Draw final path after exploration is complete
    # --------------------------------------------------------

    if current_step >= len(explored):

        if path:

            rows = [p[0] for p in path]

            cols = [p[1] for p in path]

            ax.plot(
                cols,
                rows,
                color="blue",
                linewidth=3
            )

    # --------------------------------------------------------
    # Start
    # --------------------------------------------------------

    ax.scatter(
        start[1],
        start[0],
        c="green",
        s=150,
        marker="o",
        label="Start"
    )

    # --------------------------------------------------------
    # Goal
    # --------------------------------------------------------

    ax.scatter(
        goal[1],
        goal[0],
        c="red",
        s=180,
        marker="*",
        label="Goal"
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    ax.set_title(
        f"{name} Search\n"
        f"Explored: {min(current_step, len(explored))}/"
        f"{len(explored)}"
    )

    ax.set_xlabel("Column")

    ax.set_ylabel("Row")

    ax.grid(True)

    ax.legend(
        loc="upper right"
    )


# ============================================================
# UPDATE ANIMATION
# ============================================================

def update(frame):

    global current_algorithm
    global current_step

    result = results[algorithm_names[current_algorithm]]

    explored_length = len(result["explored"])

    current_step += 1

    # Finished current algorithm
    if current_step > explored_length + 20:

        current_algorithm += 1

        current_step = 0

        # Stop after all algorithms
        if current_algorithm >= len(algorithm_names):

            animation.event_source.stop()

            return

    draw_frame()


# ============================================================
# START ANIMATION
# ============================================================

draw_frame()

animation = FuncAnimation(
    fig,
    update,
    interval=80,
    repeat=False
)

plt.tight_layout()

plt.show()
