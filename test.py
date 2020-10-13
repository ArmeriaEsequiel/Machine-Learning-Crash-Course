import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#def firstFunction():
my_feature = [1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0, 12.0]
my_label = [5.0, 8.8,  9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2]
#plt.scatter(my_features,my_label)
#plt.show()
learning_rate = 0.01 #que tan rapido se disminuye el error.
epochs = 10 #Cantidad de veces que se va a ver el bach de dataset
batch_size = 12

# Creo mi modelo Sequencial
my_model =tf.keras.models.Sequential()

#units= Dimension de salida, input_shape dimension de entrada.
my_model.add(tf.keras.layers.Dense(units=1,input_shape=(1,)))
#my_model.summary()

#Ahora configuramos el modelo para entrenar:
#1) Algoritmo para usar con learning_rate,
#2) Accuracy de las predicciones
#3) Metrics 
#my_model.compile(optimizer = tf.keras.optimizers.RMSprop(lr = learning_rate),loss = 'mean_squared_error',metrics = tf.keras.metrics.RootMeanSquaredError())

#Entreno mi modelo
#history = my_model.fit(x=my_feature,y=my_label,batch_size=batch_size,epochs=epochs)


#Vuelvo a entrenar mi modelo pero con mas epochs
# epochs = 600
# history = my_model.fit(x=my_feature,y=my_label,batch_size=batch_size,epochs=epochs)
#Al haber mas epochs, el error cuadradito se minimiza
#Despues del epoch 350, el loss convergio.

#configuro mi modelo con mayor o menor learning_rate y lo entreno
#learning_rate = 10
#epochs = 100
#my_model.compile(optimizer = tf.keras.optimizers.RMSprop(lr = learning_rate),loss = 'mean_squared_error',metrics = tf.keras.metrics.RootMeanSquaredError())
#history = my_model.fit(x=my_feature,y=my_label,batch_size=batch_size,epochs=epochs)
#En este caso tenemos un learning_rate alto, lo que hace diverger el loss

#reducimos los epoch y entrenamos
#learning_rate = 10
#epochs = 15
#my_model.compile(optimizer = tf.keras.optimizers.RMSprop(lr = learning_rate),loss = 'mean_squared_error',metrics = tf.keras.metrics.RootMeanSquaredError())
#history = my_model.fit(x=my_feature,y=my_label,batch_size=batch_size,epochs=epochs)
#En este caso vemos que el loss converge en el epoch 9.

