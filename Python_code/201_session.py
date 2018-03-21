"""
Dependencies:
tensorflow:1.1.6
"""

import tensorflow as tf

m1 = tf.constant([[2, 2]])
m2 = tf.constant([[3],
                  [3]])
dot_operation = tf.matmul(m1,m2)

print(dot_operation)  # wrong! no result

# launch the graph in a session
# method1 use session
sess = tf.Session()
result = sess.run(dot_operation)
print(result)  # Evaluate the tensor 'c'.
sess.close()

# method2 use session
with tf.Session() as sess:
    result_ = sess.run(dot_operation)
    print(result_)
