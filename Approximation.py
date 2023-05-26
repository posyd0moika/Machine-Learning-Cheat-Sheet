import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0, 10, 0.1)
x_test = np.arange(0, 10, 0.01)

h = None
N = len(x)
y_sin = np.sin(x)
y = y_sin + np.random.normal(0, 0.5, N)

K = lambda r: np.exp(-2 * r * r)        # гауссовское ядро
K = lambda r: np.abs(1 - r) * bool(r <= 1)        # треугольное ядро
K = lambda r: bool(r <= 1)        # прямоугольное ядро

ro = lambda xx, xi: np.abs(xx - xi)
w = lambda xx, xi: K(ro(xx, xi) / h)

plt.figure(figsize=(8, 8))
plot_number = 0

for h in [0.1, 0.3,0.8, 1]:
    y_test = []
    for xx in x_test:
        ww = np.array([w(xx, xi) for xi in x])
        yy = np.dot(ww, y) / sum(ww)            # формула Надарая-Ватсона
        y_test.append(yy)

    plot_number += 1
    plt.subplot(2, 2, plot_number)

    plt.scatter(x, y, color='black', s=10)
    plt.plot(x, y_sin, color='blue')
    plt.plot(x_test, y_test, color='red')
    plt.title(f"Гауссовское ядро с h = {h}")
    plt.grid()

plt.show()