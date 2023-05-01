import numpy as np
import pandas as pd
import keras

#Get the train data csv file as a pandas Dataframe
train_dataset = pd.read_csv('data/train_data.csv', header=0)
#Sort the data by Label
train_dataset = train_dataset.sort_values(by='Label')
#Change the train dataset format to a Categorical
train_dataset['Label'] = pd.Categorical(train_dataset['Label'])
#Encode the Label attribute using numerical code
train_dataset['Label'] = train_dataset.Label.cat.codes

#Get the validation data csv file as a pandas Dataframe
val_dataset = pd.read_csv('data/val_data.csv', header=0)
#Sort the dataframe by the label
val_dataset = val_dataset.sort_values(by='Label')
#Convert the dataframe into a Categorical object
val_dataset['Label'] = pd.Categorical(val_dataset['Label'])
#Encode the label attribute with numerical code
val_dataset['Label'] = val_dataset.Label.cat.codes

#Extract the label and remove the label colunm from the dataset
train_label = train_dataset.pop('Label')
#Extract the training data by copying the dataset without the label
train_data = train_dataset.copy()
#Extract the label and remove the label colunm from the dataset
val_label = val_dataset.pop('Label')
#Extract the validation data by copying the dataset without the label
val_data = val_dataset.copy()

#Format training data and validation data to numpy array
train_data = np.array(train_data)
val_data = np.array(val_data)

#Reshape the numpy array so it become feedable to a neural network
train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], 1))
val_data = np.reshape(val_data, (val_data.shape[0], val_data.shape[1], 1))

#Define the number of label
num_label = 6

#One hot matrix encoding of training label and validation label
train_label = keras.utils.to_categorical(train_label, num_label)
val_label = keras.utils.to_categorical(val_label, num_label)

#Create a plain sequential model
convNN = keras.models.Sequential()

#Adding convolutional layers
convNN.add(keras.layers.Conv1D(filters=32, kernel_size=4, padding='causal', activation='relu', input_shape=(42,1)))
convNN.add(keras.layers.MaxPooling1D(pool_size=2))
convNN.add(keras.layers.Conv1D(filters=64, kernel_size=4, padding='causal', activation='relu'))
convNN.add(keras.layers.Conv1D(filters=64, kernel_size=4, padding='causal', activation='relu'))
convNN.add(keras.layers.MaxPooling1D(pool_size=2))
convNN.add(keras.layers.Conv1D(filters=128, kernel_size=4, padding='causal', activation='relu', input_shape=(42,1)))
convNN.add(keras.layers.Conv1D(filters=128, kernel_size=4, padding='causal', activation='relu', input_shape=(42,1)))
convNN.add(keras.layers.MaxPooling1D(pool_size=2))
convNN.add(keras.layers.Conv1D(filters=256, kernel_size=4, padding='causal', activation='relu', input_shape=(42,1)))
convNN.add(keras.layers.Conv1D(filters=256, kernel_size=4, padding='causal', activation='relu', input_shape=(42,1)))
convNN.add(keras.layers.MaxPooling1D(pool_size=2))
#Leave behind 20% of the nodes based on their probability
convNN.add(keras.layers.Dropout(rate=0.2))
#Add a flatten layers in order to add Dense layer
convNN.add(keras.layers.Flatten())
#Adding Dense layer
convNN.add(keras.layers.Dense(512, activation='relu'))
convNN.add(keras.layers.Dense(256, activation='relu'))
#Output Layer with the softmax function
convNN.add(keras.layers.Dense(num_label, activation='softmax'))
#Compile the model
convNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
#Training: using train data and train label and check with the train label and val label, 100 itteration
convNN.fit(train_data, train_label, epochs=100, batch_size=32, validation_data=(val_data, val_label))