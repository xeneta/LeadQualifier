import tensorflow as tf
import csv
import numpy as np
from random import randrange

# This net is not working, as it predicts all 0's or all 1's at the moment.

# variables for the net
SEED = 3
FIRST_HIDDEN = 500
SECOND_HIDDEN = 50
FINAL_LAYER = 2
BATCH_SIZE = 100

def convertToFloat(lst):
    return np.array(lst).astype(np.float)

def fetchData(path):
    labels = []
    data = []
    f = open(path)
    csv_f = csv.reader(f)
    for row in csv_f:
        labels.append(convertToFloat(row[0]))
        data.append(convertToFloat(row[1:]))
    f.close()
    return np.array(data), np.array(labels)

def convertToOneHot(arr):
    labels = []
    for n in arr:
        if n == 0:
            labels.append([1, 0])
        elif n == 1:
            labels.append([0, 1])
    return np.array(labels)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, seed=SEED)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# create variables and placeholders for tensorflows computational graph
x = tf.placeholder(tf.float32, shape=[None, 5000])
y_ = tf.placeholder(tf.float32, shape=[None, FINAL_LAYER])

W_1 = weight_variable([5000, FIRST_HIDDEN])
b_1 = bias_variable([FIRST_HIDDEN])

W_2 = weight_variable([FIRST_HIDDEN, SECOND_HIDDEN])
b_2 = bias_variable([SECOND_HIDDEN])

W_3 = weight_variable([SECOND_HIDDEN, FINAL_LAYER])
b_3 = bias_variable([FINAL_LAYER])

hidden_layer_1 = tf.nn.sigmoid(tf.matmul(x, W_1) + b_1)
hidden_layer_2 = tf.nn.sigmoid(tf.matmul(hidden_layer_1, W_2) + b_2)
y =  tf.nn.softmax(tf.matmul(hidden_layer_2, W_3) + b_3)


# Manually calculating the loss
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Automatically calculating the loss
cross_entropy = tf.reduce_mean(
     tf.nn.softmax_cross_entropy_with_logits(y, y_)
)

# possible other loss function, if not one hot vector
#loss = tf.reduce_mean(tf.abs(tf.sub(y_, y)))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# we need to initialize all variables
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# fetch the training and testing data
X_test, y_test = fetchData('data/test.csv')
X_train, y_train = fetchData('data/train.csv')
y_test = convertToOneHot(y_test)
y_train = convertToOneHot(y_train)


# loop through the data to run the regression and update the weights
for i in range(1000):
    r = randrange(0, 1447)
    start = r
    stop = r + BATCH_SIZE
    x_train_batch = X_train[start: stop]
    y_train_batch = y_train[start: stop]
    sess.run(train_step, feed_dict={
        x: x_train_batch,
        y_: y_train_batch
    })

    if i % 100 == 0:
        cross_entropy_out = sess.run([cross_entropy], feed_dict={
            x: X_test,
            y_: y_test
        })
        print 'cross_entropy_out:', cross_entropy_out

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print 'accuracy: ', sess.run(accuracy, feed_dict={x: X_test, y_: y_test})










