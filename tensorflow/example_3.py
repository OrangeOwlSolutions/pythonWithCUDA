# --- https://www.tensorflow.org/guide/eager
#from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

# --- Setting eager mode.
tfe.enable_eager_execution()

# --- Define constant tensors.
a = tf.constant(2)
b = tf.constant(3)
print("a = %i" % a)
print("b = %i" % b)

# --- Run the operation without the session.
c = a + b
d = a * b
print("a + b = %i" % c)
print("a * b = %i" % d)

# --- Full compatibility with Numpy.
# --- Tensorflow tensor
a = tf.constant([[2., 1.],
                 [1., 0.]], dtype=tf.float32)
print("Tensor:\n a = %s" % a)
# --- Numpy array
b = np.array([[3., 0.],
              [5., 1.]], dtype=np.float32)
print("NumpyArray:\n b = %s" % b)

c = a + b
print("a + b = %s" % c)

d = tf.matmul(a, b)
print("a * b = %s" % d)

# ---Iterate through tf tensor.
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        print(a[i][j])
