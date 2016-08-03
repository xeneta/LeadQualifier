import tensorflow as tf
import csv
import numpy as np
from random import randrange

sess = tf.InteractiveSession()

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
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# fetch the training and testing data
X_test, y_test = fetchData('data/test.csv')
X_train, y_train = fetchData('data/train.csv')
y_test = convertToOneHot(y_test)
y_train = convertToOneHot(y_train)


# create variables and placeholders for tensorflows computational graph
x = tf.placeholder(tf.float32, shape=[None, 5000])
y_ = tf.placeholder(tf.float32, shape=[None, 2])
W_1 = weight_variable([5000, 500])
b_1 = bias_variable([500])
W_2 = weight_variable([500, 2])
b_2 = bias_variable([2])

hidden_layer = tf.nn.relu(tf.matmul(x, W_1) + b_1)
y = tf.nn.softmax(tf.matmul(hidden_layer, W_2) + b_2)

# we need to initialize all variables
sess.run(tf.initialize_all_variables())

# define loss function and optimizer (gradient descent)
loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

batch_size = 10

# loop through the data to run the regression and update the weights
for i in range(1000):
    r = randrange(0, 1530)
    train_x = X_train[batch_size * r: batch_size * (r + 1)]
    train_y = y_train[batch_size * r: batch_size * (r + 1)]
    train_step.run(feed_dict={x: train_x, y_: train_y})

# evaluate the model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: X_test, y_: y_test}))










