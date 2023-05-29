from algorithm import *
import matplotlib.pyplot as plt
from data import *
x = x2


plt.scatter(
    x=[i[0] for i in x],
    y=[i[1] for i in x],
    s=2
)
m, e = 4, 60
model = DBSCAN2D(m, e)
model.fit(x)
col = ["r", "purple", "g", "yellow"]
for index, item in enumerate(model.clasters):
    t = [k for k in item.claster_point]
    plt.scatter(
        x=[i[0] for i in t],
        y=[i[1] for i in t], c=col[index]
    )
    plt.title(f"M={m}, E={e}")

print(len(model.clasters))

plt.show()
print(model.clasters[0].claster_point.keys())
print(model.clasters[1].claster_point.keys())
