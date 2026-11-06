import numpy as np
import matplotlib.pyplot as plt
from collections import deque


# -------------------------------------------------
# MAZE
# 1 = Wall
# 0 = Free space
# -------------------------------------------------

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


# Start and goal
start = (1, 1)
goal = (9, 10)


# -------------------------------------------------
# MOVEMENT DIRECTIONS
# -------------------------------------------------

directions = [
    (-1, 0),   # Up
    (1, 0),    # Down
    (0, -1),   # Left
    (0, 1)     # Right
]


# -------------------------------------------------
# CHECK VALID POSITION
# -------------------------------------------------

def is_valid(position):

    row, col = position

    # Check boundaries
    if row < 0 or row >= maze.shape[0]:
        return False

    if col < 0 or col >= maze.shape[1]:
        return False

    # Check wall
    if maze[row, col] == 1:
        return False

    return True


# -------------------------------------------------
# BREADTH-FIRST SEARCH
# -------------------------------------------------

def bfs():

    # Queue for BFS
    queue = deque([start])

    # Visited cells
    visited = set()

    # Parent dictionary
    parent = {}

    # Cells explored
    explored = []

    # Mark start visited
    visited.add(start)

    while queue:

        # Remove first element
        current = queue.popleft()

        # Record exploration
        explored.append(current)

        # Check goal
        if current == goal:
            break

        # Explore neighbors
        for direction in directions:

            new_position = (
                current[0] + direction[0],
                current[1] + direction[1]
            )

            # Check if valid and not visited
            if is_valid(new_position) and new_position not in visited:

                # Mark visited
                visited.add(new_position)

                # Store parent
                parent[new_position] = current

                # Add to queue
                queue.append(new_position)

    # -------------------------------------------------
    # RECONSTRUCT PATH
    # -------------------------------------------------

    path = []

    if goal in visited:

        current = goal

        while current != start:

            path.append(current)

            current = parent[current]

        path.append(start)

        # Reverse path
        path.reverse()

    return path, explored


# -------------------------------------------------
# RUN BFS
# -------------------------------------------------

path, explored = bfs()


# -------------------------------------------------
# PRINT RESULTS
# -------------------------------------------------

print("================================")
print("       BFS MAZE SOLVER")
print("================================")

print("Start:", start)
print("Goal :", goal)

print("Cells explored:", len(explored))

if path:

    print("Goal reached!")
    print("Path length:", len(path))

    print("Path:")
    print(path)

else:

    print("Goal cannot be reached.")


# -------------------------------------------------
# VISUALIZATION
# -------------------------------------------------

plt.figure(figsize=(8, 6))

# Show maze
plt.imshow(maze, cmap="gray_r")


# -------------------------------------------------
# Show explored cells
# -------------------------------------------------

for row, col in explored:

    if (row, col) != start and (row, col) != goal:

        plt.scatter(
            col,
            row,
            c="yellow",
            s=100
        )


# -------------------------------------------------
# Show shortest path
# -------------------------------------------------

if path:

    path_rows = [p[0] for p in path]
    path_cols = [p[1] for p in path]

    plt.plot(
        path_cols,
        path_rows,
        color="blue",
        linewidth=3,
        label="BFS Shortest Path"
    )


# -------------------------------------------------
# Start
# -------------------------------------------------

plt.scatter(
    start[1],
    start[0],
    c="green",
    s=200,
    marker="o",
    label="Start"
)


# -------------------------------------------------
# Goal
# -------------------------------------------------

plt.scatter(
    goal[1],
    goal[0],
    c="red",
    s=200,
    marker="*",
    label="Goal"
)


# -------------------------------------------------
# Formatting
# -------------------------------------------------

plt.title("BFS Maze Solving")

plt.xlabel("Column")
plt.ylabel("Row")

plt.grid(True)

plt.legend()

plt.show()