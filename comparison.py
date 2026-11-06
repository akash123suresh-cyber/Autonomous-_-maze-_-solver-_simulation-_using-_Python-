import numpy as np
import time
import heapq
from collections import deque
import matplotlib.pyplot as plt


# ============================================================
# MAZE
# 1 = WALL
# 0 = FREE SPACE
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


# ============================================================
# START AND GOAL
# ============================================================

start = (1, 1)
goal = (19, 29)


# ============================================================
# MOVEMENT
# ============================================================

directions = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1)
]


# ============================================================
# VALID POSITION
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

    path = []

    if goal in visited:

        current = goal

        while current != start:

            path.append(current)

            current = parent[current]

        path.append(start)

        path.reverse()

    return path, explored


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

    path = []

    if goal in visited:

        current = goal

        while current != start:

            path.append(current)

            current = parent[current]

        path.append(start)

        path.reverse()

    return path, explored


# ============================================================
# A*
# ============================================================

def heuristic(position):

    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])


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

                h = heuristic(neighbor)

                f = new_g + h

                heapq.heappush(
                    open_list,
                    (f, neighbor)
                )

    path = []

    if goal in g_cost:

        current = goal

        while current != start:

            path.append(current)

            current = parent[current]

        path.append(start)

        path.reverse()

    return path, explored


# ============================================================
# RUN AND MEASURE
# ============================================================

algorithms = {
    "DFS": dfs,
    "BFS": bfs,
    "A*": astar
}

results = {}


for name, algorithm in algorithms.items():

    start_time = time.perf_counter()

    path, explored = algorithm()

    end_time = time.perf_counter()

    results[name] = {
        "path": path,
        "explored": explored,
        "path_length": len(path),
        "time": end_time - start_time
    }


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("              MAZE SEARCH COMPARISON")
print("=" * 70)

print(
    f"{'Algorithm':<15}"
    f"{'Explored':<15}"
    f"{'Path Length':<15}"
    f"{'Time (s)':<15}"
)

print("-" * 70)

for algorithm_name, result in results.items():
    print(
        f"{algorithm_name:<15}"
        f"{len(result['explored']):<15}"
        f"{result['path_length']:<15}"
        f"{result['time']:<15.8f}"
    )
print("=" * 70)

print()
print("Start:", start)
print("Goal :", goal)


# ============================================================
# VISUALIZE THREE ALGORITHMS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 7))

for ax, (name, result) in zip(axes, results.items()):

    ax.imshow(maze, cmap="gray_r")

    # Explored cells
    for row, col in result["explored"]:

        if (row, col) != start and (row, col) != goal:

            ax.scatter(
                col,
                row,
                c="yellow",
                s=20
            )

    # Path
    path = result["path"]

    if path:

        rows = [p[0] for p in path]
        cols = [p[1] for p in path]

        ax.plot(
            cols,
            rows,
            color="blue",
            linewidth=2
        )

    # Start
    ax.scatter(
        start[1],
        start[0],
        c="green",
        s=100,
        marker="o"
    )

    # Goal
    ax.scatter(
        goal[1],
        goal[0],
        c="red",
        s=120,
        marker="*"
    )

    ax.set_title(
        f"{name}\n"
        f"Explored: {result['explored'].__len__()} | "
        f"Path: {result['path_length']}"
    )

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    ax.grid(True)

plt.suptitle(
    "DFS vs BFS vs A*",
    fontsize=16
)

plt.tight_layout()

plt.show()
