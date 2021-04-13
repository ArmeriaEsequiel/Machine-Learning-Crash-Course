import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NUMERIC_COLUMNS = ['Cylinders','Displacement','Horsepower','Weight','Acceleration','Model year']
CATEGORICAL_COLUMNS = ['Origin']
columns_names = ['Mpg']+NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

##Load data
not_split_data = pd.read_csv('auto-mpg.csv',na_values='?',names = columns_names, comment='\t', sep=' ', skipinitialspace=True)
not_split_data = not_split_data.dropna()
not_split_data['Origin'] = not_split_data['Origin'].map({1:'Usa', 2:'Europe', 3:'Japan'})

##Divide data into train and eval:
df_train = not_split_data.sample(frac = 0.8,random_state = 0) #80% of the data is to train.
df_eval = not_split_data.drop(df_train.index) #Drop rows used to train. so test dont repeat any value on evaluation.


## Evaluate data:

#not_split_data.plot(kind = 'scatter',x = 'Cylinders', y = 'Mpg')
#not_split_data.plot(kind = 'scatter',x = 'Acceleration', y = 'Mpg')
#not_split_data.plot(kind = 'scatter', x ='Origin', y= 'Mpg')
#not_split_data.plot(kind = 'scatter', x = 'Cylinders', y = 'Mpg')
#not_split_data.plot(kind = 'scatter', x = 'Weight', y = 'Mpg')
#not_split_data.plot(kind = 'scatter', x = 'Model year', y = 'Mpg')
#not_split_data.plot(kind = 'scatter', x = 'Model year', y = 'Cylinders')
#plt.show()

# From this data analisys, we can note that
# 1) Japani's cars has higher mpg of all 3 coutries.
# 2) Cars with less cylinders, has higher mpg
# 3) Cars with less weith, has higher mpg
# 4) Newer cars, has higher mpg.


##Prepare dataframe:
#Pop label column from both dataframes.
train_l = df_train.pop('Mpg')
eval_l = df_eval.pop('Mpg')

#Get categorical features from dataframe
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
	vocabulary = df_train[feature_name].unique() #Get uniques categorical values
	feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name,vocabulary))

#Get numeric features from dataframe
for feature_name in NUMERIC_COLUMNS:
	feature_columns.append(tf.feature_column.numeric_column(feature_name))

#Make input function to create a dataset to feed the model
def input_fn(df_data, df_label,epochs, training, batch_size):
	ds = tf.data.Dataset.from_tensor_slices((dict(df_data),df_label)) #create dataset
	if training:
		ds = ds.shuffle(1000) #if training, shuffle data
	ds = ds.batch(batch_size).repeat(epochs) #split data into batch of batch_size and repeat for numbers of epochs
	return(ds)


# Use gradient descent as the optimizer for training the model.
my_optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)

##Creating the model
# We use linearRegressor. Because we want to predict a value, the mpg value.
regressor_model= tf.estimator.LinearRegressor(feature_columns = feature_columns, optimizer = my_optimizer)

## Train our model:
regressor_model.train(input_fn =lambda: input_fn(df_train,train_l,epochs = 2150, training = True, batch_size = 150))


eval_result = regressor_model.evaluate(input_fn = lambda: input_fn(df_eval, eval_l, epochs = 1, training = False, batch_size = 150))

print(eval_result)
#print("El accuracy es {}".format(eval_result['accuracy']))

