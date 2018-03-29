import tensorflow as tf
a = tf.constant(2)
b = tf.constant(3)
with tf.Session() as sess:
    print("a=2, b=3")
    print("Addition with constants: %i" % sess.run(a+b))
    print("Multiplication with constants: %i" % sess.run(a*b))

#placeholder 相当于分配空内存，feed_dict将数据放入内存内
a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)
add = tf.add(a,b) #一个张量， 保存的是数字计算过程
mul = tf.multiply(a,b)
with tf.Session() as sess:
    print('%i' % sess.run(add, feed_dict={a:10,b:30}))
    print('%i'% sess.run(mul, feed_dict={a:10,b:30}))

#开eager模式， 会变成命令行模式，不需要session.run()
'''
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
'''