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

inp = Input((2,))
out = Dense(2, activation="sigmoid")(inp)
model = keras.Model(inp, out)
model.summary()

opt = RMSprop(learning_rate=0.1)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50)
# categorical_crossentropy
# binary_crossentropy

temp = [[(i, j) for i in range(50)] for j in range(75)]
temp = np.array(temp)
temp.shape = (-1, 2,)

result = model.predict(temp)

res = temp[[i for i in range(len(result)) if 0.475 < result[i][0] < 0.525]]

x_0 = x_train[y == 1]
x_1 = x_train[y == 0]

plt.scatter(x_0[:, 0], x_0[:, 1], color='red')
plt.scatter(x_1[:, 0], x_1[:, 1], color='blue')
plt.plot(res[:, 0], res[:, 1], c='y')
plt.show()