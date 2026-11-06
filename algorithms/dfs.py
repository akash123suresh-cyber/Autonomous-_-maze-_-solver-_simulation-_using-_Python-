import numpy as np
import matplotlib.pyplot as plt

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

# Starting and goal positions
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
# CHECK WHETHER A CELL CAN BE VISITED
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
# DEPTH-FIRST SEARCH
# -------------------------------------------------

def dfs():

    # Stack stores cells that need to be explored
    stack = [start]

    # Store visited cells
    visited = set()

    # Store parent of each cell
    parent = {}

    # Store order in which cells were explored
    explored = []

    # Mark starting cell as visited
    visited.add(start)

    while stack:

        # Take the last element from stack
        current = stack.pop()

        # Record explored cell
        explored.append(current)

        # Check if goal reached
        if current == goal:
            break

        # Check all possible movements
        for direction in directions:

            new_position = (
                current[0] + direction[0],
                current[1] + direction[1]
            )

            # Check whether new cell is valid
            if is_valid(new_position) and new_position not in visited:

                # Mark as visited
                visited.add(new_position)

                # Remember where we came from
                parent[new_position] = current

                # Add to stack
                stack.append(new_position)

    # -------------------------------------------------
    # RECONSTRUCT PATH
    # -------------------------------------------------

    path = []

    # If goal was reached
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
# RUN DFS
# -------------------------------------------------

path, explored = dfs()


# -------------------------------------------------
# PRINT RESULTS
# -------------------------------------------------

print("================================")
print("       DFS MAZE SOLVER")
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

# Display maze
plt.imshow(maze, cmap="gray_r")

# -------------------------------------------------
# Show explored cells
# -------------------------------------------------

for row, col in explored:

    # Don't overwrite start/goal
    if (row, col) != start and (row, col) != goal:

        plt.scatter(
            col,
            row,
            c="yellow",
            s=100
        )


# -------------------------------------------------
# Show final path
# -------------------------------------------------

if path:

    path_rows = [p[0] for p in path]
    path_cols = [p[1] for p in path]

    plt.plot(
        path_cols,
        path_rows,
        color="blue",
        linewidth=3,
        label="DFS Path"
    )


# -------------------------------------------------
# Start position
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
# Goal position
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

plt.title("DFS Maze Solving")

plt.xlabel("Column")
plt.ylabel("Row")

plt.grid(True)

plt.legend()

plt.show()