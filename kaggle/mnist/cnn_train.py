# -*- coding: utf-8 -*-
import tensorflow as tf
import os 
import pandas as pd
import numpy as np
#超参数
INPUT_NODE = 784
OUTPUT_NODE = 10
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10
#第一层
CONV1_DEEP = 32
CONV1_SIZE = 5
# 第二层
CONV2_DEEP = 64
CONV2_SIZE = 5
# FC
FC_SIZE = 512

#CNN 前向传播
def inference(input_tensor, train, regularizer):
    #卷积层1 28*28*1 -> 28*28*32
    X = tf.reshape(input_tensor, shape=[-1,28,28,1])
    with tf.variable_scope('layer1-conv1'): #5*5*32过滤器
        conv1_weights = tf.get_variable("weight", 
            [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], 
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_bias = tf.get_variable("bias", [CONV1_DEEP],  
            initializer=tf.constant_initializer(0.0))
        #strides步长为1， padding全0填充
        conv1 = tf.nn.conv2d(X, conv1_weights, strides=[1,1,1,1], padding = 'SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))

    #池化层1 28*28*32 -> 14*14*32
    #name_scope 是给op_name加前缀, variable_scope是给get_variable()创建的变量的名字加前缀。
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    #卷积层2 14*14*32 -> 14*14*64
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable("weight",
            [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_bias = tf.get_variable("bias", [CONV2_DEEP],
            initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1,1,1,1], padding = 'SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))

    #池化层2 14*14*64 -> 7*7*64
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    ##输入FC前reshape shape为 batch_size*7*7*64 pool_shape[0]为batch_size
    pool_shape = pool2.get_shape().as_list() 
    print(pool_shape)
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    #FC1 49*64拉直， 用dropout避免过拟合
    with tf.variable_scope("layer5-fc1"):
        fc1_weights = tf.get_variable("weight", 
            [nodes, FC_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            #可以认为这里的regularizer是个函数指针
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_bias = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.0))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_bias)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5) #dropout一般只在fc层使用

    with tf.variable_scope("layer6-fc2"):
        fc2_weights = tf.get_variable("weight", 
            [FC_SIZE, NUM_LABELS], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            #可以认为这里的regularizer是个函数指针
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_bias = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.0))

        logit = tf.matmul(fc1, fc2_weights) + fc2_bias        
    return logit


REGULARAZTION_RATE = 0.0001
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99 #滑动平均， 减少过拟合
BATCH_SIZE = 100
MODEL_SAVE_PATH = "D:/kaggle/mnist/"
MODEL_NAME = "model.ckpt"
def train(mnist):
    x = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], name = 'x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name= 'y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

    y = inference(x, True, regularizer)
    global_step = tf.Variable(0, trainable=False)
    df = mnist['label']
    data_np = mnist.as_matrix()
    data_y = pd.get_dummies(mnist['label']).as_matrix() #number->one hot 
    data_x = data_np[:,1:] #pixel
    print("total data_num=%d" % data_y.shape[0])
    num_examples = data_y.shape[0]
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')


    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        
        index_in_epoch = 0
        for i in range(TRAINING_STEPS):
            #batch decent
            start = index_in_epoch
            index_in_epoch += BATCH_SIZE
            if index_in_epoch > num_examples:
                perm = np.arange(num_examples)
                np.random.shuffle(perm)
                data_x = data_x[perm]
                data_y = data_y[perm]
                start = 0
                index_in_epoch = BATCH_SIZE
                assert BATCH_SIZE <= num_examples
            end = index_in_epoch
            xs = np.reshape(data_x[start:end], (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
             
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: data_y[start:end]})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    data_train = pd.read_csv('D:/kaggle/mnist/train.csv')
    #data_np = data_train.as_matrix()
    #pixel[0].size 784 nparray 
    train(data_train)

if __name__ == '__main__':
    tf.app.run()