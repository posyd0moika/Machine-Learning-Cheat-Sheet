import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
y = -1
x_ = np.array([[10, 50], [20, 30], [25, 30], [20, 60], [15, 70], [40, 40], [30, 45], [20, 45], [40, 30], [7, 35]])
y_ = np.array([y, 1, 1, y, y, 1, 1, y, 1, y])
x_train = tf.constant(x_, dtype=tf.float32)
y_train = tf.constant(y_, dtype=tf.float32)

opt = tf.optimizers.SGD(learning_rate=0.01)

w = tf.Variable(tf.random.normal((2,)), dtype=tf.float32, name="w")
b = tf.Variable(0.0, dtype=tf.float32, name="b")
# print(x_train,y_train,w,b)

ep = 500
lr = 0.001
losss = []
i = 0
s = 0
for i in range(ep):
    print(f"{i}/{ep}")
    for x, y in zip(x_train, y_train):
        with tf.GradientTape() as tape:
            M = (x[0] * w[0] + x[1]*w[1] + b) * -y
            loss = tf.nn.sigmoid(-M) #- 20 * lr * tf.reduce_mean(tf.square(w))
            # loss = tf.square(y + loss)

        grad = tape.gradient(loss,[w,b])
        w = tf.Variable(w - grad[0] * lr)
        b = tf.Variable(b - grad[1] * lr)
        # opt.apply_gradients((grad[0],w))
        # opt.apply_gradients((grad[1],b))


        s = float(loss)
        i += 1
    losss.append(s)

plt.plot(losss)
plt.show()

line_y = [x * float(w[0]) / float(w[1]) + float(b) for x in range(50)]
plt.plot(line_y)
x_0 = x_[y_ == 1]
x_1 = x_[y_ == y]
plt.scatter(x_0[:, 0], x_0[:, 1], color='red')
plt.scatter(x_1[:, 0], x_1[:, 1], color='blue')
plt.show()
print(losss[-1])
