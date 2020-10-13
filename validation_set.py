import numpy as np
import pandas as pd
import tensorflow as tf 
import matplotlib as plt
#Hypeparameters
learning_rate = 0.08
epochs = 30
batch_size = 100

#Split the original train_df into a reducen train_df and a validation set.
validation_split = 0.2


#Cantidad de filas para imprimir por pantalla.(None imprime todo el df)
pd.options.display.max_rows = 10


pd.options.display.float_format = "{:.1f}".format
train_df = pd.read_csv('california_housing_train.csv')#leemos data set para entrenar
test_df = pd.read_csv('california_housing_test.csv')#leemos data set para testear
scale_factor = 1000.0


# Scale the training set's label.
train_df["median_house_value"] /= scale_factor 

# Scale the test set's label
test_df["median_house_value"] /= scale_factor
#print(test_df)


#creamos el modelo
my_model = tf.keras.models.Sequential()

#Dimension de la salida y entrada
my_model.add(tf.keras.layers.Dense(units=1,input_shape=(1,)))

#Ahora configuramos el modelo para entrenar:
#1) Algoritmo para usar con learning_rate,
#2) Accuracy de las predicciones
#3) Metrics 
my_model.compile(optimizer = tf.keras.optimizers.RMSprop(lr = learning_rate),loss = 'mean_squared_error',metrics = tf.keras.metrics.RootMeanSquaredError())




#Entrenamos el modelo para predecir el valor de las casas dependiendo
# del ingreso medio de los vecinos del barrio.
my_feature = 'median_income'
my_label = 'median_house_value'
#history = my_model.fit(x = train_df[my_feature],y= train_df[my_label], 
#						batch_size=batch_size, epochs=epochs,
#						validation_split =valdation_split)


#Reajustamos el modelo para obtener un loss de entrenamiento y validacion cercanos.
#Permutando las columnas del df en el que vamos a entrenar
shuffled_train_df = train_df.reindex(np.random.permutation(train_df.index))

epochs = 80

#Vuelvo a entrenar con el nuevo df para entrenar
history = my_model.fit(x = shuffled_train_df[my_feature],y= shuffled_train_df[my_label], 
						batch_size=batch_size, epochs=epochs,
						validation_split =validation_split)
#Testeamos el modelo
x_test = test_df[my_feature]
y_test = test_df[my_label]
result = my_model.evaluate(x_test,y_test,batch_size=batch_size)

print('-----------------------------------------------------')
root_test_error = result[1]
val_error = history.history['val_root_mean_squared_error'][-1]
root_error = history.history['root_mean_squared_error'][-1]

#Vemos que los errores son muy cercanos.
print('eror en test {}',format(root_test_error))
print('eror en train {}',format(root_error))
print('eror en validate{}',format(val_error))
