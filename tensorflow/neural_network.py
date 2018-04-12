"""
FC neural-network
2 hidden-layer 每层256个神经元
"""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("D:\mnist", one_hot=True)

import tensorflow as tf

learning_rate = 0.1
num_steps = 500
batch_size = 128
disp_step = 100

h1_num = 256
h2_num = 256 #hidden layer neuron num
num_input = 784 
num_classes = 10

X = tf.placeholder("float", [None,num_input])
Y = tf.placeholder("float",[None, num_classes])
weights = {
    'h1':tf.Variable(tf.random_normal([num_input, h1_num])),
    'h2':tf.Variable(tf.random_normal([h1_num, h2_num])),
    'out':tf.Variable(tf.random_normal([h2_num, num_classes]))
}
bias = {
    'b1':tf.Variable(tf.random_normal([h1_num])),
    'b2':tf.Variable(tf.random_normal([h2_num])),
    'out':tf.Variable(tf.random_normal([num_classes]))
}

def neural_net(x):
    layer1 = tf.add(tf.matmul(x, weights['h1']), bias['b1'])
    layer2 = tf.add(tf.matmul(layer1, weights['h2']), bias['b2'])
    out_layer = tf.add(tf.matmul(layer2, weights['out']), bias['out'])
    return out_layer

y_ = neural_net(X)
prediction = tf.nn.softmax(y_)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % disp_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss_disp, acc = sess.run([loss, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss_disp) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
    print("optimization finish")

    print("test accuracy:", sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))

