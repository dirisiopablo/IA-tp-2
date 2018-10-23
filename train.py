# Importo la lib tensorflow para entrenar el modelo
import tensorflow as tf
# Importo otras libs
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

CONST_LEARN_RATE = 0.3
CONST_EPOCHS = 400

# Leo los ratings del .dat
ratings = pd.read_csv('dataset/ratings.dat', sep="::", header = None, engine='python')
# Pivoteo la data para tener la info por usuario
# Normalizo para ver si devuelve valores más exactos
ratings_pivot = pd.pivot_table(ratings[[0,1,2]], values=2, index=0, columns=1 ).fillna(0) / 5
# Creo los sets de entrenamiento y testeo
X_train, X_test = train_test_split(ratings_pivot, train_size=0.75)

# Decido la cantidad de nodos para las capas
n_nodes_inpl = 3706
n_nodes_hl1  = 256
n_nodes_outl = 3706
# La capa oculta tiene 784*32 weights y 32 biases
hidden_1_layer_vals = {'weights':tf.Variable(tf.random_normal([n_nodes_inpl+1,n_nodes_hl1]))}
# La capa output tiene 784*32 weights y 32 biases
output_layer_vals = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1+1,n_nodes_outl])) }

# Capa input tiene 3706 ratings por usuario
input_layer = tf.placeholder('float', [None, 3706], name='input_layer')
# Agrego un nodo constante a la input capa con la misma shape para concatenar luego
input_layer_const = tf.fill( [tf.shape(input_layer)[0], 1], 1.0)
input_layer_concat =  tf.concat([input_layer, input_layer_const], 1)
# Multiplico el output de la input layer con el peso de la matrix
layer_1 = tf.nn.sigmoid(tf.matmul(input_layer_concat, hidden_1_layer_vals['weights']))
# Agrego un bias node a la capa oculta
layer1_const = tf.fill( [tf.shape(layer_1)[0], 1], 1.0)
layer_concat =  tf.concat([layer_1, layer1_const], 1)
# Multiplico la salida de la capa oculta con el peso de la matrix para tener el output final
# Le aplicamos la sigmoidal para que dé valores entre 0 y 1
output_layer = tf.nn.sigmoid(tf.matmul( layer_concat,output_layer_vals['weights']), name='output_layer')
# output_true tiene que tener la misma shape original para calcular los errores
output_true = tf.placeholder('float', [None, 3706])

# Defino la funcion de costo
error =    tf.reduce_mean(tf.square(output_layer - output_true))

# ratio de entrenamiento, que tan rapido entrena
learn_rate = CONST_LEARN_RATE

# Defino el optimizador
optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(error)

# Inicializo las variables y abro la sesion
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
<<<<<<< HEAD
# Defino el tamaño del batch, la cantidad de épocas y el ratio de aprendizaje
batch_size = 100  
hm_epochs = CONST_EPOCHS
tot_images = X_train.shape[0]
=======
# Defino el tamaño del batch, la cantidad de epocas y el ratio de aprendizaje
batch_size = 100
hm_epochs = 200
tot_images = X_train.shape[0]
>>>>>>> 7b6d2a4f0573fcdeaddb0bbc1273ef312b412d40

# Entreno el modelo con las 200 epocas tomando los usuarios a batches de 100
# La mejora se imprime al final de cada epoca
print('\nEntrenando modelo...\n')
for epoch in range(hm_epochs):
    epoch_loss = 0 # Inicializo el error en 0
    batchs = 0
    
    for i in range(int(tot_images/batch_size)):
        epoch_x = X_train[ i*batch_size : (i+1)*batch_size ]
        _, c = sess.run([optimizer, error], feed_dict={input_layer: epoch_x, output_true: epoch_x})
        epoch_loss += c
        batchs =  i + 1

    epoch_loss /= batchs # Error promedio por batch por época
    output_train = sess.run(output_layer, feed_dict={input_layer:X_train})
    output_test = sess.run(output_layer, feed_dict={input_layer:X_test})
        
    print('MSE train', MSE(output_train, X_train),'MSE test', MSE(output_test, X_test))
    print('Epoca', epoch, '/', hm_epochs, 'perdida:',epoch_loss)

# Guardo el modelo entrenado
print('\nSe guarda el modelo entrenado...')
cwd = os.getcwd()
path = os.path.join(cwd, 'model')
shutil.rmtree(path, ignore_errors=True)
X_test_tf = tf.convert_to_tensor(X_test, name="X_test")
X_train_tf = tf.convert_to_tensor(X_train, name="X_train")
saver = tf.train.Saver()
save_path = saver.save(sess, path + "/model", global_step=1000)
print('Ok')
print("Modelo guardado en path: %s" % save_path)