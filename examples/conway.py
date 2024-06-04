import itertools
import numpy as np
import neunet
import neunet.nn as nn
import neunet.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from tqdm import tqdm

"""
Conway's Game of Life

This example illustrates how to implement a neural network that can be trained to simulate Conway's Game of Life.
"""

N = 128
# Randomly create a grid
# grid = np.random.binomial(1, p = 0.2, size = (N, N))

# or define for example the Glider Gun configuration as shown in
# https://conwaylife.com/wiki/Gosper_glider_gun
# Other examples can be found in
# https://conwaylife.com/patterns/

grid = np.zeros((N, N))

gun_pattern_src = """
........................O...........
......................O.O...........
............OO......OO............OO
...........O...O....OO............OO
OO........O.....O...OO..............
OO........O...O.OO....O.O...........
..........O.....O.......O...........
...........O...O....................
............OO......................
"""

# Split the pattern into lines
lines = gun_pattern_src.strip().split("\n")

# Convert each line into an array of 1s and 0s
gun_pattern_grid = np.array(
    [[1 if char == "O" else 0 for char in line] for line in lines]
)

grid[0 : gun_pattern_grid.shape[0], 0 : gun_pattern_grid.shape[1]] = gun_pattern_grid


def update(grid):
    """
    Native implementation of Conway's Game of Life
    """
    updated_grid = grid.copy()
    for i in range(N):
        for j in range(N):
            # Use the modulo operator % to ensure that the indices wrap around the grid.
            # Using the modulus operator % to index an array creates the effect of a "toroidal" mesh, which can be thought of as the surface of a donut
            n_alived_neighbors = int(
                grid[(i - 1) % N, (j - 1) % N]
                + grid[(i - 1) % N, j]
                + grid[(i - 1) % N, (j + 1) % N]
                + grid[i, (j - 1) % N]
                + grid[i, (j + 1) % N]
                + grid[(i + 1) % N, (j - 1) % N]
                + grid[(i + 1) % N, j]
                + grid[(i + 1) % N, (j + 1) % N]
            )

            if grid[i, j] == 1:
                if n_alived_neighbors < 2 or n_alived_neighbors > 3:
                    updated_grid[i, j] = 0
            else:
                if n_alived_neighbors == 3:
                    updated_grid[i, j] = 1

    return updated_grid


class GameOfLife(nn.Module):
    def __init__(
        self,
    ):
        super(GameOfLife, self).__init__()

        self.conv = nn.Conv2d(1, 1, 3, padding=0, bias=False)
        kernel = neunet.tensor([[[[1, 1, 1], [1, 0, 1], [1, 1, 1]]]])
        self.conv.weight.data = kernel.data

    def forward(self, grid: np.ndarray):
        """
        Implementation of Conway's Game of Life using a convolution (works much faster)
        """
        # Pad the grid to create a "toroidal" mesh effect
        grid_tensor = neunet.tensor(np.pad(grid, pad_width=1, mode="wrap"))[
            None, None, :, :
        ]
        n_alive_neighbors = self.conv(grid_tensor).data
        updated_grid = (n_alive_neighbors.astype(int) == 3) | (
            (grid.astype(int) == 1) & (n_alive_neighbors.astype(int) == 2)
        )
        updated_grid = updated_grid[0, 0, :, :]

        return updated_grid


game = GameOfLife()


class Dataset:
    # def __init__(self, grid_size):
    #     self.grid_size = grid_size

    # def get_data(self, n_samples = 100000):
    #     '''
    #     Generate data using the game
    #     '''
    #     X, Y = [], []

    #     for _ in tqdm(range(n_samples), desc = 'Generating Dataset'):
    #         x = np.random.binomial(1, p = 0.2, size = (self.grid_size, self.grid_size))
    #         y = game(x)

    #         X.append(x)
    #         Y.append(y)

    #     return np.array(X), np.array(Y)

    def get_data(self):
        """
        Generate data from all probable situations (2^9),
        where (1 point - current point, 8 points - surrounding neighbors points)
        """
        X = list(itertools.product([0, 1], repeat=9))

        X = [np.array(x).reshape(3, 3) for x in X]
        Y = [game(x).astype(int) for x in X]

        return np.array(X), np.array(Y)


# architecture was borrowed from https://gist.github.com/failure-to-thrive/61048f3407836cc91ab1430eb8e342d9
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3, padding=0)  # 2
        self.conv2 = nn.Conv2d(10, 1, 1)

    def forward(self, x):
        x = neunet.tanh(self.conv1(x))
        x = self.conv2(x)
        return x

    def predict(self, x):
        # Pad the grid to create a "toroidal" mesh effect
        x = neunet.tensor(np.pad(x, pad_width=1, mode="wrap"))[None, None, :, :]
        # Squeeze
        return self.forward(x).data[0, 0, :, :]


model = Net()

dataset = Dataset()
X, Y = dataset.get_data()

optimizer = optim.Adam(model.parameters(), lr=0.01)
# optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.MSELoss()

epochs = 500

for epoch in range(epochs):
    tqdm_range = tqdm(zip(X, Y), total=len(X))
    perm = np.random.permutation(len(X))

    X = X[perm]
    Y = Y[perm]
    losses = []
    for x, y in tqdm_range:
        optimizer.zero_grad()

        x = neunet.tensor(np.pad(x, pad_width=1, mode="wrap"))[None, None, :, :]
        y = neunet.tensor(y)[None, None, :, :]
        y_pred = model(x)

        loss = criterion(y_pred, y)

        loss.backward()
        optimizer.step()
        losses.append(loss.data)
        tqdm_range.set_description(
            f"Epoch: {epoch + 1}/{epochs}, Loss: {loss.data:.7f}, Mean Loss: {np.mean(losses):.7f}"
        )

model.eval()


def animate(i):
    global grid
    ax.clear()
    # grid = update(grid) # Native implementation
    # grid = game(grid) # Implementation using convolution
    grid = model.predict(grid)  # Neural network
    ax.imshow(grid, cmap=ListedColormap(["black", "lime"]))  # , interpolation='lanczos'


fig, ax = plt.subplots(figsize=(10, 10))

ani = animation.FuncAnimation(fig, animate, frames=30, interval=5)

plt.show()
