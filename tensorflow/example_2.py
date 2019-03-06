# --- https://www.tensorflow.org/api_docs/python/tf/placeholder

import tensorflow as tf

# --- Addition and multiplication with constants.
a = tf.constant(5)
b = tf.constant(6)

sess1 = tf.Session()
print("Addition with constants       : %i" % sess1.run(a + b))
print("Multiplication with constants : %i" % sess1.run(a * b))

# --- Addition and multiplication with placeholders.
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)
mul = tf.multiply(a, b)

sess2 = tf.Session()
print("Addition with variables       : %i" % sess2.run(add, feed_dict={a: 5, b: 6}))
print("Multiplication with variables : %i" % sess2.run(mul, feed_dict={a: 5, b: 6}))

# --- Matrix-matrix multiplication.
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
product = tf.matmul(matrix1, matrix2)

sess3 = tf.Session()
print(sess3.run(product))

