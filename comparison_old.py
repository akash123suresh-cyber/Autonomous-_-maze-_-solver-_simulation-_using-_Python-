import numpy as np
import time
import heapq
from collections import deque


# =================================================
# MAZE
# =================================================

maze = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,1,0,0,0,0,0,0,1],
    [1,0,1,0,1,0,1,1,1,1,0,1],
    [1,0,1,0,0,0,0,0,0,1,0,1],
    [1,0,1,1,1,1,1,1,0,1,0,1],
    [1,0,0,0,0,0,0,1,0,1,0,1],
    [1,1,1,1,1,1,0,1,0,1,0,1],
    [1,0,0,0,0,1,0,0,0,1,0,1],
    [1,0,1,1,0,1,1,1,1,1,0,1],
    [1,0,0,1,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1]
])


start = (1, 1)
goal = (9, 10)


# =================================================
# MOVEMENT
# =================================================

directions = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1)
]


# =================================================
# VALID CELL
# =================================================

def is_valid(position):

    row, col = position

    if row < 0 or row >= maze.shape[0]:
        return False

    if col < 0 or col >= maze.shape[1]:
        return False

    if maze[row, col] == 1:
        return False

    return True


# =================================================
# DFS
# =================================================

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


# =================================================
# BFS
# =================================================

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


# =================================================
# A*
# =================================================

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


# =================================================
# RUN ALGORITHMS
# =================================================

algorithms = {
    "DFS": dfs,
    "BFS": bfs,
    "A*": astar
}


results = []


for name, algorithm in algorithms.items():

    start_time = time.perf_counter()

    path, explored = algorithm()

    end_time = time.perf_counter()

    execution_time = end_time - start_time

    results.append({
        "algorithm": name,
        "explored": len(explored),
        "path_length": len(path),
        "time": execution_time,
        "success": len(path) > 0
    })


# =================================================
# DISPLAY RESULTS
# =================================================

print()
print("=" * 65)
print("             SEARCH ALGORITHM COMPARISON")
print("=" * 65)

print(
    f"{'Algorithm':<15}"
    f"{'Explored':<15}"
    f"{'Path Length':<15}"
    f"{'Time (s)':<15}"
)

print("-" * 65)


for result in results:

    print(
        f"{result['algorithm']:<15}"
        f"{result['explored']:<15}"
        f"{result['path_length']:<15}"
        f"{result['time']:<15.8f}"
    )


print("=" * 65)

print()
print("Start:", start)
print("Goal :", goal)

print()
print("All algorithms completed.")