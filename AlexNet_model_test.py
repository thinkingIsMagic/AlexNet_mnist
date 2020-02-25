import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random

# --------1.加载数据集，定义网络超参数--------
mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
train_num = mnist.train.images.shape[0] #训练集样本个数
learning_rate = 0.001 #学习率
batch_size = 128 #mini—batch大小
training_iters = 5 * train_num/batch_size #迭代次数,5个训练样本集
display_step = 10 #显示间隔
n_input = 784 #输入维度
n_classes = 10 #输出维度
dropout = 0.75 #dropout大小
x = tf.placeholder(tf.float32,[None,n_input])
y = tf.placeholder(tf.float32,[None,n_classes])
keep_prob = tf.placeholder(tf.float32)

# --------2.定义网络结构--------
def conv2d(name, x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1,strides,strides,1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x, name=name)
def maxpool2d(name, x ,k=2):
    return tf.nn.max_pool(x, ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME', name=name)
def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001/9.0, beta=0.75, name=name)

weights = {'wc1': tf.Variable(tf.random_normal([11,11,1,96])),
           'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256])),
           'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
           'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
           'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
           'wd1': tf.Variable(tf.random_normal([4 * 4 * 256, 1024])),
           'wd2': tf.Variable(tf.random_normal([1024, 200])),
           'out': tf.Variable(tf.random_normal([200, n_classes]))}
biases  = {'bc1': tf.Variable(tf.random_normal([96])),
           'bc2': tf.Variable(tf.random_normal([256])),
           'bc3': tf.Variable(tf.random_normal([384])),
           'bc4': tf.Variable(tf.random_normal([384])),
           'bc5': tf.Variable(tf.random_normal([256])),
           'bd1': tf.Variable(tf.random_normal([1024])),
           'bd2': tf.Variable(tf.random_normal([200])),
           'out': tf.Variable(tf.random_normal([n_classes]))}

def alex_net(x,weights,biases,dropout):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    #卷积层，5次卷积，3次池化，2次局部响应归一化
    conv1 = conv2d('conv1', x, weights['wc1'], biases['bc1'])
    pool1 = maxpool2d('pool1', conv1)
    norm1 = norm('norm1', pool1 )
    conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'])
    pool2 = maxpool2d('pool2', conv2)
    norm2 = norm('norm2', pool2)
    conv3 = conv2d('conv3', norm2, weights['wc3'], biases['bc3'])
    conv4 = conv2d('conv4', conv3, weights['wc4'], biases['bc4'])
    conv5 = conv2d('conv5', conv4, weights['wc5'], biases['bc5'])
    pool3 = maxpool2d('pool5', conv5)
    #全连接层, 两层全连接，输出接softmax
    fc1 = tf.reshape(pool3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.nn.relu(tf.matmul(fc1, weights['wd1']) + biases['bd1'])
    fc1 = tf.nn.dropout(fc1, dropout)
    fc2 = tf.nn.relu(tf.matmul(fc1,weights['wd2']) + biases['bd2'])
    fc2 = tf.nn.dropout(fc2, dropout)
    out = tf.matmul(fc2, weights['out']) + biases['out']
    return out

# --------3.检测--------
out = alex_net(x, weights, biases, keep_prob) #alexnet网络前向传播
init = tf.global_variables_initializer()
saver = tf.train.Saver() #保存模型
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, 'model/model.ckpt')
    print('Model restored.')
    for i in range(10):
        index = random.randint(0,9999) #0-9999随便选数字
        prediction = tf.argmax(out, 1)
        pred = sess.run(prediction, feed_dict={x: [mnist.test.images[index]],
                                               keep_prob: 1.0})
        label = np.argmax(mnist.test.labels[index])
        #if label != pred[0]:
        print('实际:', label, ',识别:', pred[0])
        image = np.reshape(mnist.test.images[index],[28,28])
        plt.imshow(image, cmap=plt.get_cmap('gray_r'))
        plt.show()