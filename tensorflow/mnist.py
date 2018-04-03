from __future__ import print_function
import tensorflow as tf
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("D:\mnist", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
hx = tf.nn.softmax(tf.add(tf.matmul(x, W), b))
init = tf.global_variables_initializer()
#cost function
#如果多个类别中，各个类别是可以相互包含的则使用logistic回归，如果是互斥的则使用softmax回归
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hx), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

'''
旧写法
#softmax的交叉熵
hx = tf.nn.softmax(tf.add(tf.multiply(W,x),b))
cost = tf.reduce_mean(- tf.reduce_sum(y*tf.log(hx), reduction_indices=1))
#reduction_indices =1 代表横向压缩矩阵， 最终可以压缩成一个数

#sigmoid方法， 用于二分类
hx = tf.nn.softmax(tf.add(tf.multiply(W,x), b))
loss = tf.reduce_mean(- y * tf.log(hx) - (1 - y * tf.log(1 - hx))

'''
with tf.Session() as sess:
    sess.run(init)

    for i in range(training_epochs): #训练轮数
        batch_num = int(mnist.train.num_examples / batch_size)
        avg_cost = 0
        for j in range(batch_num):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x:batch_x, y:batch_y})

            avg_cost += c/batch_size

        print("Epoch:", '%04d' % (i+1), "cost=", "{:.9f}".format(avg_cost))

    print("train finish")

    correct_prediction = tf.equal(tf.argmax(hx, 1), tf.argmax(y, 1)) #argmax 返回数组最大的值的下标
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #bool转化成float32
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))