import matplotlib.pyplot as plt
import numpy as np

import neunet
from neunet.optim import SGD, Adadelta, Adagrad, Adam, Adamax, Momentum, NAdam, RMSprop


def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def himmelblau(x, y):
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


def matyas(x, y):
    return 0.26 * (x**2 + y**2) - 0.48 * x * y


def beale(x, y):
    return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y**2) ** 2 + (2.625 - x + x * y**3) ** 2


def booth(x, y):
    return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2


def goldstein_price(x, y):
    return (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)) * (
        30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2)
    )


def gradient_descent(starting_point, optimizer, num_iterations, function, learning_rate):
    points = [starting_point]
    point = neunet.tensor(starting_point, requires_grad=True)
    optimizer = optimizer(lr=learning_rate, params=[point])
    for _ in range(num_iterations):
        optimizer.zero_grad()
        y = function(point[0], point[1])
        y.backward()
        optimizer.step()
        points.append([point[0].detach().numpy(), point[1].detach().numpy()])
    return np.array(points)


starting_point = np.array([-0.0, -7.0])  # (x, y)
learning_rate = 0.01
num_iterations = 1000
optimizers = [Adam, SGD, RMSprop, NAdam, Momentum, Adagrad, Adadelta, Adamax]
function = himmelblau

x = np.linspace(-10, 10, 1000)
y = np.linspace(-10, 10, 1000)
X, Y = np.meshgrid(x, y)
Z = function(X, Y)


# 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, alpha=0.5, rstride=30, cstride=30, cmap="viridis")

for i, optimizer in enumerate(optimizers):
    points = gradient_descent(starting_point, optimizer, num_iterations, function, learning_rate)
    ax.plot(*points.T, function(*points.T), label=optimizer.__name__, color=f"C{i}")

plt.legend(loc="upper left")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

# Contour plot
plt.contourf(
    X, Y, Z, levels=500, extent=[0, 5, 0, 5], origin="lower", cmap="coolwarm"
)  # cmaps = ['coolwarm', 'RdYlBu', 'viridis', 'jet', 'RdGy']
plt.colorbar()
contours = plt.contour(X, Y, Z, levels=10, extent=[0, 5, 0, 5], colors="black")
plt.clabel(contours, inline=True, fontsize=8)

for i, optimizer in enumerate(optimizers):
    points = gradient_descent(starting_point, optimizer, num_iterations, function, learning_rate)
    plt.plot(points[:, 0], points[:, 1], label=optimizer.__name__, color=f"C{i}")

plt.legend(loc="upper left")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
