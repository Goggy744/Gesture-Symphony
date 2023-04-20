import numpy as np
import pandas as pd
import keras


train_dataset = pd.read_csv('data/train_data.csv', header=0)
train_dataset = train_dataset.sort_values(by='Label')
train_dataset['Label'] = pd.Categorical(train_dataset['Label'])
train_dataset['Label'] = train_dataset.Label.cat.codes

val_dataset = pd.read_csv('data/val_data.csv', header=0)
val_dataset = val_dataset.sort_values(by='Label')
val_dataset['Label'] = pd.Categorical(val_dataset['Label'])
val_dataset['Label'] = val_dataset.Label.cat.codes

train_label = train_dataset.pop('Label')
train_data = train_dataset.copy()

val_label = val_dataset.pop('Label')
val_data = val_dataset.copy()

train_data = np.array(train_data)
val_data = np.array(val_data)

train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], 1))
val_data = np.reshape(val_data, (val_data.shape[0], val_data.shape[1], 1))

num_label = 6

train_label = keras.utils.to_categorical(train_label, num_label)
val_label = keras.utils.to_categorical(val_label, num_label)

convNN = keras.models.Sequential()

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
convNN.add(keras.layers.Dropout(rate=0.2))
convNN.add(keras.layers.Flatten())
convNN.add(keras.layers.Dense(512, activation='relu'))
convNN.add(keras.layers.Dense(256, activation='relu'))
convNN.add(keras.layers.Dense(num_label, activation='softmax'))

convNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

convNN.fit(train_data, train_label, epochs=100, batch_size=32, validation_data=(val_data, val_label))