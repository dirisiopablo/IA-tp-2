# Importing tensorflow
import tensorflow as tf
# Importing some more libraries
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

# reading the ratings data
ratings = pd.read_csv('dataset/ratings.dat', sep="::", header = None, engine='python')
# Lets pivot the data to get it at a user level
ratings_pivot = pd.pivot_table(ratings[[0,1,2]], values=2, index=0, columns=1 ).fillna(0) / 5 #normalizo para ver si devuelve valores más exactos
# creating train and test sets
X_train, X_test = train_test_split(ratings_pivot, train_size=0.8)

# Deciding how many nodes wach layer should have
n_nodes_inpl = 3706  
n_nodes_hl1  = 256  
n_nodes_outl = 3706  
# first hidden layer has 784*32 weights and 32 biases
hidden_1_layer_vals = {'weights':tf.Variable(tf.random_normal([n_nodes_inpl+1,n_nodes_hl1]))}
# first hidden layer has 784*32 weights and 32 biases
output_layer_vals = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1+1,n_nodes_outl])) }

# user with 3706 ratings goes in
input_layer = tf.placeholder('float', [None, 3706], name='input_layer')
# add a constant node to the first layer
# it needs to have the same shape as the input layer for me to be
# able to concatinate it later
input_layer_const = tf.fill( [tf.shape(input_layer)[0], 1] ,1.0  )
input_layer_concat =  tf.concat([input_layer, input_layer_const], 1)
# multiply output of input_layer wth a weight matrix 
layer_1 = tf.nn.sigmoid(tf.matmul(input_layer_concat, hidden_1_layer_vals['weights']))
# adding one bias node to the hidden layer
layer1_const = tf.fill( [tf.shape(layer_1)[0], 1] ,1.0  )
layer_concat =  tf.concat([layer_1, layer1_const], 1)
# multiply output of hidden with a weight matrix to get final output
output_layer = tf.nn.sigmoid(tf.matmul( layer_concat,output_layer_vals['weights']), name='output_layer') # le aplicamos la sigmoidal para que dé valores entre 0 y 1
# output_true shall have the original shape for error calculations
output_true = tf.placeholder('float', [None, 3706])
# define our cost function
error =    tf.reduce_mean(tf.square(output_layer - output_true))
# define our optimizer
learn_rate = 0.1   # how fast the model should learn
optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(error)

# initialising variables and starting the session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# defining batch size, number of epochs and learning rate
batch_size = 100  # how many images to use together for training
hm_epochs = 200  # how many times to go through the entire dataset
tot_images = X_train.shape[0] # total number of images

# running the model for a 200 epochs taking 100 users in batches
# total improvement is printed out after each epoch
print('\nTraining...')
for epoch in range(hm_epochs):
    epoch_loss = 0    # initializing error as 0
    batchs = 0
    
    for i in range(int(tot_images/batch_size)):
        epoch_x = X_train[ i*batch_size : (i+1)*batch_size ]
        _, c = sess.run([optimizer, error],
                        feed_dict={input_layer: epoch_x,
                        output_true: epoch_x})
        epoch_loss += c
        batchs =  i + 1

    epoch_loss /= batchs # error promedio por batch por época
        
    output_train = sess.run(output_layer,
               feed_dict={input_layer:X_train})

    output_test = sess.run(output_layer,
                   feed_dict={input_layer:X_test})
        
    print('MSE train', MSE(output_train, X_train),'MSE test', MSE(output_test, X_test))      
    print('Epoch', epoch, '/', hm_epochs, 'loss:',epoch_loss)

# Save model state
print('\nSaving trained model...')
cwd = os.getcwd()
path = os.path.join(cwd, 'model')
shutil.rmtree(path, ignore_errors=True)
X_test_tf = tf.convert_to_tensor(X_test, name="X_test")
X_train_tf = tf.convert_to_tensor(X_train, name="X_train")
saver = tf.train.Saver()
save_path = saver.save(sess, path + "/model", global_step=1000)
print('Ok')
print("Model saved in path: %s" % save_path)