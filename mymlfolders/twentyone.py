# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 22:19:49 2018

@author: NANCUH
"""

import tensorflow as tf
n_node_hl1 = 2
n_classes = 2
xx = [[1.0,1.0],[1.0,0.0],[0.0,1.0],[0.0,0.0]]
yy = {0.0,1.0,1.0,0.0}
def neural_network_model(data):
    hidden_1_layer = {'weight':tf.Variable(tf.random_normal([2,n_node_hl1])),'biases':tf.Variable(tf.random_normal([n_node_hl1]))}
    output_layer = {'weight':tf.Variable(tf.random_normal([n_node_hl1,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}
    #(input_data * weights)+biases
    
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']),hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    output = tf.add(tf.matmul(l1,output_layer['weight']),output_layer['biases'])
    return output
def train_neural_network(xx):
    prediction = neural_network_model(xx)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=yy))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    #cycles feed forward + backprop
    hm_epochs = len(xx)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(hm_epochs):
                epoch_x,epoch_y = xx,yy
                _,c = sess.run([optimizer,cost],feed_dict= {x:epoch_x,y:epoch_y})
                epoch_loss+=c
            print('Epoch',epoch,'completed out of',hm_epochs,'loss:', epoch_loss)
        
        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_max(tf.cast(correct,'float'))
        print("accuracy:",accuracy.eval({x:xx,y:yy}))
    
train_neural_network(xx)