import numpy as np
import matplotlib.pyplot as plt
import heapq


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


# -------------------------------------------------
# START AND GOAL
# -------------------------------------------------

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

    if row < 0 or row >= maze.shape[0]:
        return False

    if col < 0 or col >= maze.shape[1]:
        return False

    if maze[row, col] == 1:
        return False

    return True


# -------------------------------------------------
# MANHATTAN DISTANCE
# -------------------------------------------------

def heuristic(position):

    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])


# -------------------------------------------------
# A* ALGORITHM
# -------------------------------------------------

def astar():

    # Priority queue
    # Stores: (f_cost, position)
    open_list = []

    # Start node
    heapq.heappush(
        open_list,
        (heuristic(start), start)
    )

    # Cost from start to each cell
    g_cost = {
        start: 0
    }

    # Parent of each cell
    parent = {}

    # Cells explored
    explored = []

    # Keep track of visited/processed cells
    closed_set = set()

    while open_list:

        # Get cell with lowest f-cost
        current_f, current = heapq.heappop(open_list)

        # Ignore if already processed
        if current in closed_set:
            continue

        # Mark as processed
        closed_set.add(current)

        # Record exploration
        explored.append(current)

        # Check goal
        if current == goal:
            break

        # Explore neighbors
        for direction in directions:

            neighbor = (
                current[0] + direction[0],
                current[1] + direction[1]
            )

            # Ignore walls and invalid cells
            if not is_valid(neighbor):
                continue

            # Cost to reach neighbor
            new_g_cost = g_cost[current] + 1

            # If this is a better path
            if neighbor not in g_cost or new_g_cost < g_cost[neighbor]:

                # Update cost
                g_cost[neighbor] = new_g_cost

                # Remember parent
                parent[neighbor] = current

                # Calculate heuristic
                h_cost = heuristic(neighbor)

                # Total cost
                f_cost = new_g_cost + h_cost

                # Add to priority queue
                heapq.heappush(
                    open_list,
                    (f_cost, neighbor)
                )

    # -------------------------------------------------
    # RECONSTRUCT PATH
    # -------------------------------------------------

    path = []

    if goal in g_cost:

        current = goal

        while current != start:

            path.append(current)

            current = parent[current]

        path.append(start)

        path.reverse()

    return path, explored


# -------------------------------------------------
# RUN A*
# -------------------------------------------------

path, explored = astar()


# -------------------------------------------------
# PRINT RESULTS
# -------------------------------------------------

print("================================")
print("       A* MAZE SOLVER")
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
        label="A* Path"
    )


# -------------------------------------------------
# START
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
# GOAL
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
# FORMATTING
# -------------------------------------------------

plt.title("A* Maze Solving")

plt.xlabel("Column")
plt.ylabel("Row")

plt.grid(True)

plt.legend()

plt.show()