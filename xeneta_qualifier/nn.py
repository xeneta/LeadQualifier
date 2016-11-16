import tensorflow as tf
import csv
import numpy as np
from random import randrange
SEED = 1

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

# fetch the training and testing data
X_test, y_test = fetchData('data/test.csv')
X_train, y_train = fetchData('data/train.csv')
y_test = convertToOneHot(y_test)
y_train = convertToOneHot(y_train)

print 'len(X_train): ', len(X_train)
print 'len(X_test): ', len(X_test)

FIRST_HIDDEN = 2000
SECOND_HIDDEN = 500
THIRD_HIDDEN = 50
# create variables and placeholders for tensorflows computational graph
x = tf.placeholder(tf.float32, shape=[None, 5000])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

W_1 = weight_variable([5000, FIRST_HIDDEN])
b_1 = bias_variable([FIRST_HIDDEN])

W_2 = weight_variable([FIRST_HIDDEN, SECOND_HIDDEN])
b_2 = bias_variable([SECOND_HIDDEN])

W_3 = weight_variable([SECOND_HIDDEN, THIRD_HIDDEN])
b_3 = bias_variable([THIRD_HIDDEN])

W_4 = weight_variable([THIRD_HIDDEN, 2])
b_4 = bias_variable([2])

hidden_layer = tf.nn.sigmoid(tf.matmul(x, W_1) + b_1)
hidden_layer_2 = tf.nn.sigmoid(tf.matmul(hidden_layer, W_2) + b_2)
hidden_layer_3 = tf.nn.sigmoid(tf.matmul(hidden_layer_2, W_3) + b_3)

#y = tf.nn.softmax(tf.matmul(hidden_layer, W_2) + b_2)
y = tf.matmul(hidden_layer_3, W_4) + b_4


# we need to initialize all variables

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


# define loss function and optimizer (gradient descent)
log_y = tf.log(y)
log_calc = y_ * log_y
cross = -tf.reduce_sum(log_calc, reduction_indices=[1])
#cross_entropy = tf.reduce_mean(cross)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, y_)
cross_mean = tf.reduce_mean(cross_entropy)

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

batch_size = 50


# loop through the data to run the regression and update the weights
for i in range(500):
    r = randrange(0, 1530)
    start = r
    stop = r + batch_size
    x_train_batch = X_train[start: stop]
    y_train_batch = y_train[start: stop]
    _, _cross_entropy, _cross, _log_calc, _log_y, _y = sess.run([train_step, cross_entropy, cross, log_calc, log_y, y], feed_dict={
        x: x_train_batch,
        y_: y_train_batch
    })
    #print '_cross_entropy = %s' % _cross_entropy
    #print '_cross = %s' % _cross
    #print '_y_ = %s' % _y_
    #print '_y = %s' % _y


    if i % 100 == 0:
        current_loss, _cross_mean = sess.run([cross_entropy, cross_mean], feed_dict={
            x: X_test,
            y_: y_test
        })

        #print 'current_loss: ', current_loss
        print '_cross_mean', _cross_mean
        #print 'current_b: ',current_b
        #print '----------'

# evaluate the model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(session=sess,feed_dict={x: X_test, y_: y_test}))










