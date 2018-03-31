"""
Dependencies
tensorflow: 1.1.6
matplotlib
numpy
"""

import tensorflow as tf
import matplotlib as plt
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

# fake data
n_data = np.ones((100,2))
x0 = np.random.normal(2*n_data,1)  # # class0 x shape = (100,2)  # 从mean为2，标准差为1的正态分布随机选取数值填充到shape中各个元素位置
y0 = np.zeros(100)
x1 = np.random.normal(-2*n_data,1)
y1 = np.ones(100)
x = np.vstack((x0,x1))     # shape(200,2)
y = np.hstack((y0,y1))    # shape(1,200)

# plot data
plt.scatter(x[:, 0], x[:, 1], c=y, s=100, lw=0,cmp='RDYlGn')  # x[:,0] :第一列，x[:,1] 第二列

tf_x = tf.placeholder(tf.float.32, x.shape)  # input x
tf_y = tf.placeholder(tf.float.32, y.shape)  # input y

# neural network layers
l1 = tf.layers.dense(tf_x, 10, tf.relu)  # hidden layer
output = tf.layers.dense(l1, 2)

loss = tf.losses.spare_softmax_cross_entropy(labels=tf_y, logits=output)
accuracy = tf.metrics.accuracy(
    labels = tf.squeeze(tf_y), predictions = tf.argmax(output, axis=1),)[1]
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train_op = optimizer.minimize(loss)

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.loacl_variables_initializer())
sess.run(init_op)    # initializer var in graph

plt.ion()

for step in range(100):
    # train and net output
    _, acc, pred = sess.run([train_op, accuracy, output], {tf_x: x, tf_y: y})
    if step % 2 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x[:,0], x[:,1], c=pred.argmax(1), s=100, lw=0, cmap = 'RdYlGn')
        plt.text(1.5, -4, 'Accuracy=%.2f' %acc, fontdict = {'size':20, 'color':'red'})
        plt.pause(0.1)
 plt.ioff()
 plt.show()
