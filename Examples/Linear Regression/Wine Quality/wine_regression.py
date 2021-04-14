import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


##Load data from csv
COLUMNS_NAMES = ['Fixed acidity','Volatile acidity','Citric acid','Residual sugar',
				'Chlorides','Free sulfur dioxide','Total silfur dioxide','Density',
				'pH','Sulphates','Alcohol','Quality']

not_split_data = pd.read_csv('winequality-white.csv', names = COLUMNS_NAMES, header =0)


# Make some feature seleccion:
correlations = not_split_data.corr()
sns.heatmap(correlations, annot = True,cmap = plt.cm.CMRmap_r)
#plt.show()




##Divide data into train and eval:

df_train =  not_split_data.sample(frac=0.8, random_state = 0) # 80% of data is for training the model
df_eval = not_split_data.drop(df_train.index) #Drop rows used to train. so test dont repeat any value on evaluation.

#Analyzing data:
#not_split_data.plot(kind = 'scatter',x = 'Alcohol', y = 'Quality')
#not_split_data.plot(kind = 'scatter',x = 'Quality', y = 'pH')
#not_split_data.plot(kind = 'scatter', x ='Sulphates', y= 'Quality')
#not_split_data.plot(kind = 'scatter', x = 'Residual sugar', y = 'Quality')
#not_split_data.plot(kind = 'scatter', x = 'Free sulfur dioxide', y = 'Quality')
#plt.show()





# Pop label column from train and eval dataframe
train_l = df_train.pop('Quality')
eval_l = df_eval.pop('Quality')

# Create the model with Keras:
my_model = tf.keras.Sequential()

## Add layers
n_features = len(COLUMNS_NAMES) - 1 # Get features

# units = 1, only one neuron wich return one value
# input_shape = n_features, data that the model will work with
my_model.add(tf.keras.layers.Dense(units = 1, input_shape =(n_features,)))


# Configure the model to train
my_model.compile(optimizer = tf.keras.optimizers.RMSprop(lr = 0.001), loss = 'mean_squared_error',
					metrics = tf.keras.metrics.RootMeanSquaredError())

# Train the model:
epochs = 300
validation_split = 0.2
batch_size = 100

#history = my_model.fit(x = df_train, y = train_l, epochs = epochs, batch_size = batch_size, validation_split = validation_split)
#print(history.history.keys())

# Evaluate model

#result = my_model.evaluate(df_eval, eval_l, batch_size = batch_size)
#print(result)

#print('-----------------------------------------------------')
#root_test_error = result[1]
#val_error = history.history['val_root_mean_squared_error'][-1]
#root_error = history.history['root_mean_squared_error'][-1]

#Vemos que los errores son muy cercanos.
#print('eror en test {}',format(root_test_error))
#print('eror en train {}',format(root_error))
#print('eror en validate{}',format(val_error))
#test_predictions = my_model.predict(df_eval)
#plt.scatter(eval_l,test_predictions)
#plt.xlabel('True Values')
#plt.ylabel('Predictions')
#plt.xlim([0,plt.xlim()][1])
#plt.ylim([0,plt.ylim()][1])
#_ = plt.plot([-100,100],[-100,100])
#plt.show()