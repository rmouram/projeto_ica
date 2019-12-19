# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 12:45:21 2019

@author: romul
"""

import tensorflow as tf
import numpy as np
import math
from sklearn.model_selection import train_test_split
from tensorflow.python.framework import ops
ops.reset_default_graph()

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

def data_encode(file):
	X = []
	Y = []
	train_file = open(file, 'r')
	for line in train_file.read().strip().split('\n'):
		line = line.split(',')
		Y.append(line[0])
		X.append([line[1],line[2],line[3],line[4],line[5],line[6],line[7],line[8],line[9],line[10],line[11],line[12],line[13]])
	return X,Y

file = "winetrain2.txt"
data, target = data_encode(file)
data = np.asarray(data)
target = np.asarray(target)

N_INSTANCES = data.shape[0]
N_INPUT = data.shape[1]
N_CLASSES = 3
TEST_SIZE = 0.1
TRAIN_SIZE = int(N_INSTANCES * (1 - TEST_SIZE))
batch_size = 40
training_epochs = 5000
learning_rate = 0.3
display_step = 100
hidden_size = 5

target_ = np.zeros((N_INSTANCES, N_CLASSES))

data_train, data_test, target_train, target_test = train_test_split(data, target_, test_size=0.1, random_state=100)

x_data = tf.placeholder(shape=[None, N_INPUT], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, N_CLASSES], dtype=tf.float32)

# creates activation function
def gaussian_function(input_layer):
    initial = math.exp(-2*math.pow(input_layer, 2))
    return initial

np_gaussian_function = np.vectorize(gaussian_function)

def d_gaussian_function(input_layer):
    initial = -4 * input_layer * math.exp(-2*math.pow(input_layer, 2))
    return initial

np_d_gaussian_function = np.vectorize(d_gaussian_function)

np_d_gaussian_function_32 = lambda input_layer: np_d_gaussian_function(input_layer).astype(np.float32)

def tf_d_gaussian_function(input_layer, name=None):
    with ops.name_scope(name, "d_gaussian_function", [input_layer]) as name:
        y = tf.py_func(np_d_gaussian_function_32, [input_layer],[tf.float32], name=name, stateful=False)
    return y[0]

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    rnd_name = 'PyFunGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def gaussian_function_grad(op, grad):
    input_variable = op.inputs[0]
    n_gr = tf_d_gaussian_function(input_variable)
    return grad * n_gr

np_gaussian_function_32 = lambda input_layer: np_gaussian_function(input_layer).astype(np.float32)

def tf_gaussian_function(input_layer, name=None):
    with ops.name_scope(name, "gaussian_function", [input_layer]) as name:
        y = py_func(np_gaussian_function_32, [input_layer], [tf.float32], name=name, grad=gaussian_function_grad)
    return y[0]
# end of defining activation function

def rbf_network(input_layer, weights):
    layer1 = tf.matmul(tf_gaussian_function(input_layer), weights['h1'])
    layer2 = tf.matmul(tf_gaussian_function(layer1), weights['h2'])
    output = tf.matmul(tf_gaussian_function(layer2), weights['output'])
    return output

weights = {
    'h1': tf.Variable(tf.random_normal([N_INPUT, hidden_size])),
    'h2': tf.Variable(tf.random_normal([hidden_size, hidden_size])),
    'output': tf.Variable(tf.random_normal([hidden_size, N_CLASSES]))
}

pred = rbf_network(x_data, weights)
print(pred)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_target))
print(cost)
my_opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)
print(my_opt)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_target, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	for epoch in range(training_epochs):
		_, c= sess.run([my_opt,cost],feed_dict = {x_data: data_train, y_target: target_train})
		if(epoch + 1) % display_step == 0:
			print("Epoch:",epoch+1,"Cost:", c)
	print("Optimization Finished")

	test_result = sess.run(pred,feed_dict = {x_data: data_train})
	correct_prediction = tf.equal(tf.argmax(test_result,1),tf.argmax(target_train,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

	print("accuracy:", accuracy.eval({x_data: data_test, y_target: target_test}))

