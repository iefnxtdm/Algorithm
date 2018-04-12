import tensorflow as tf
import numpy as np

in1 = tf.constant([1.0,2.0,3.0], name="in1")
in2 = tf.Variable(tf.random_uniform([3]), name = 'in2')
output = tf.add_n([in1, in2], name = 'add')

write= tf.summary.FileWriter("D:/kaggle/mnist/", tf.get_default_graph())
write.close()

#运行 tensorboard --logdir=...