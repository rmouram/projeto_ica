# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 10:05:22 2019

@author: romul
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 18:14:44 2019

@author: romul
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

def label_encode(label):
	val = []
	if label == '1':
		val = [1,0,0]
	elif label == '2':
		val = [0,1,0]
	elif label == '3':
		val = [0,0,1]
	return val

def data_encode(file):
	X = []
	Y = []
	train_file = open(file, 'r')
	for line in train_file.read().strip().split('\n'):
		line = line.split(',')
		Y.append(label_encode(line[0]))
		X.append([line[1],line[2],line[3],line[4],line[5],line[6],line[7],line[8],line[9],line[10],line[11],line[12],line[13]])
	return X,Y

file = "winetrain.txt"
train_X, train_Y = data_encode(file)

#parametros da rede
learning_rate = 0.3
training_epochs = 5000
display_steps = 100

n_input = 13 #quantos valores de entrada?
n_hidden = 2 #quantos neurônios na camada oculta?
n_output = 3 #quantos neuronios na camada de saida?

#a partir daqui construimos o modelo
X = tf.placeholder("float",[None,n_input])
Y = tf.placeholder("float",[None,n_output])

weights = {
	"hidden": tf.Variable(tf.random_normal([n_input,n_hidden])),
	"output": tf.Variable(tf.random_normal([n_hidden,n_output])),
}

bias = {
	"hidden": tf.Variable(tf.random_normal([n_hidden])),
	"output": tf.Variable(tf.random_normal([n_output])),
}

def model(X, weights, bias):
	layer1 = tf.add(tf.matmul(X, weights["hidden"]),bias["hidden"])
	layer1 = tf.nn.sigmoid(layer1)
	output_layer = tf.nn.sigmoid(tf.matmul(layer1,weights["output"]) + bias["output"])
    
	return output_layer

#train_X, train_Y = data_encode("iristrain.txt") #dataset de treinamento
test_X, test_Y = data_encode("winetest.txt") #dataset de validacao

pred = model(X,weights,bias)

cost = tf.reduce_mean(tf.square(pred - Y))
optimizador = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	for epochs in range(training_epochs):
		_, c= sess.run([optimizador,cost],feed_dict = {X: train_X, Y: train_Y})
		if(epochs + 1) % display_steps == 0:
			print("Epoch:",epochs+1,"Cost:", c)
	print("Optimization Finished")

	test_result = sess.run(pred,feed_dict = {X: train_X})
	correct_prediction = tf.equal(tf.argmax(test_result,1),tf.argmax(train_Y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

	print("accuracy:", accuracy.eval({X: test_X, Y: test_Y}))