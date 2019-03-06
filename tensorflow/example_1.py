# --- https://www.tensorflow.org/api_docs/python/tf/constant
# --- https://www.tensorflow.org/api_docs/python/tf/Session

import tensorflow as tf

# --- Constant 1-D Tensor populated with value list.
tensor1 = tf.constant([1, 2, 3, 4, 5, 6, 7])

# --- Constant 2-D tensor populated with scalar value -1.
tensor2 = tf.constant(-1.0, shape=[2, 3])

# --- Start tensorflow session.
sess = tf.Session()

# --- Evaluate and print the tensors.
print(sess.run(tensor1))
print(sess.run(tensor2))
