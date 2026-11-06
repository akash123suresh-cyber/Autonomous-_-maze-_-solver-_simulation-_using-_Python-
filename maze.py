import numpy as np
import matplotlib.pyplot as plt


# 1 = wall
# 0 = free space

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


# Starting position
start = (1, 1)

# Goal position
goal = (9, 10)
# Robot can move UP, DOWN, LEFT, RIGHT

directions = [
    (-1, 0),   # UP
    (1, 0),    # DOWN
    (0, -1),   # LEFT
    (0, 1)     # RIGHT
]
def is_valid(position):

    row, col = position

    # Check whether position is inside maze
    if row < 0 or row >= maze.shape[0]:
        return False

    if col < 0 or col >= maze.shape[1]:
        return False

    # Check whether it is a wall
    if maze[row, col] == 1:
        return False

    return True

print(is_valid((1, 2)))
print(is_valid((0, 0)))
current_position = start

print("Robot starting position:", current_position)

for direction in directions:

    new_position = (
        current_position[0] + direction[0],
        current_position[1] + direction[1]
    )

    if is_valid(new_position):
        print("Valid movement:", new_position)
    else:
        print("Blocked:", new_position)

# Display maze
plt.imshow(maze, cmap="gray_r")

# Mark start
plt.scatter(start[1], start[0], c="green", s=150, label="Start")

# Mark goal
plt.scatter(goal[1], goal[0], c="red", s=150, label="Goal")

plt.title("Autonomous Maze Robot")
plt.legend()
plt.grid(True)

plt.show()
