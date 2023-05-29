import keras
import tensorflow.keras
from tensorflow.keras.layers import Input, Dense
import numpy as np
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt

x_train = np.array([
    [10, 55], [20, 30], [25, 30], [20, 60], [15, 70], [40, 40], [30, 45], [20, 45], [40, 30], [7, 35], [10, 30]
])
y = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0])
y_train = keras.utils.to_categorical(y, num_classes=2)

model = keras.Sequential([
    Input((2,)),
    Dense(200, activation="relu"),
    Dense(300, activation="relu"),
    Dense(2, activation="sigmoid")
])
model.summary()

opt = RMSprop(learning_rate=0.1)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50)
# categorical_crossentropy
# binary_crossentropy

temp = [[(i, j) for i in range(50)] for j in range(75)]
temp = np.array(temp)
temp.shape = (-1, 2,)

result = model.predict(temp)

res = temp[[i for i in range(len(result)) if 0.4 < result[i][0] < 0.6]]

x_0 = x_train[y == 1]
x_1 = x_train[y == 0]

plt.scatter(x_0[:, 0], x_0[:, 1], color='red')
plt.scatter(x_1[:, 0], x_1[:, 1], color='blue')
plt.plot(res[:, 0], res[:, 1], c='y')
plt.show()