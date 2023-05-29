from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import numpy as np
import matplotlib.pyplot as plt
import data
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay


def get_grid(data):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    return np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 1))


# plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')

x_cl1 = data.x2_cl1
x_cl2 = data.x2_cl2
cm_bright = ListedColormap(["#FF0000", "#0000FF"])
y1 = [1 for i in range(len(x_cl1))]
y2 = [0 for i in range(len(x_cl2))]

x = np.array(x_cl1 + x_cl2)
y = np.array(y1 + y2)

cl1_r = x[y == 1]
cl2_b = x[y == 0]

model = RandomForestClassifier(n_estimators=100, max_depth=100)
# model = AdaBoostClassifier(n_estimators=100)
model.fit(x, y)

xx, yy = get_grid(x)
predicted = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

cm = plt.cm.RdBu
ax = plt.subplot(1, 1, 1)
DecisionBoundaryDisplay.from_estimator(
    model, x, cmap=cm, alpha=0.8, ax=ax, eps=0.1
)

ax.scatter(cl1_r[:, 0], cl1_r[:, 1], c="r",
            edgecolors="k",
            alpha=0.6,
            )

ax.scatter(
    cl2_b[:, 0], cl2_b[:, 1],
    edgecolors="k",
    alpha=0.6,
)

plt.show()
