import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pandas import read_csv

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# load the data
# IMPORTANT: THE FILE household_power_consumption.csv MUST BE IMPORTED MANUALLY (go the README...)
csv_directory = "../household_power_consumption.csv";
dataset = read_csv(csv_directory)

# Print the data
print("======== Dataset Shape ========")
print(dataset.shape)

print("======== Dataset Head ========")
print(dataset.head())


# This function returns two windows:
# 1.
# 2.
def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i + target_size])
    return np.array(data), np.array(labels)


# The first 1.500.000 lines of the dataset for train
# The rest 507.259 lines for validation
train_split = 1500000

# Setting seed to ensure reproducibility.
tf.random.set_seed(13)

# Forecast a univariate time series!
# Extract the "Global_active_power" from the dataset
univariate_dataset = dataset['Global_active_power']
univariate_dataset.index = dataset['datetime']
univariate_dataset.head()

# Print what extracted
print(univariate_dataset)

# Observe how this data looks across time
univariate_dataset.plot(subplots=True)

# Data normalization
univariate_dataset = univariate_dataset.values
tf.keras.utils.normalize(univariate_dataset)

# The model will be given the last 20 recorded Global_Active_Power observations,
# and needs to learn to predict the Global_Active_Power at the next time step.
univariate_past_history = 20
univariate_future_target = 0

x_train_uni, y_train_uni = univariate_data(univariate_dataset,
                                           0,
                                           train_split,
                                           univariate_past_history,
                                           univariate_future_target)

x_val_uni, y_val_uni = univariate_data(univariate_dataset,
                                       train_split,
                                       None,
                                       univariate_past_history,
                                       univariate_future_target)

# Print what univariate_data calculated for training
print('Single window of past history')
print(x_train_uni[0])
print('\n Target Global_Active_Power to predict')
print(y_train_uni[0])


def create_time_steps(length):
    return list(range(-length, 0))


def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                     label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel('Time-Step')
    return plt


sample_plot = show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example')
sample_plot.show()


# Baseline Prediction
def baseline(history):
    return np.mean(history)


baseline_plot = show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0, 'Baseline Prediction Example')
baseline_plot.show()

# Recurrent Neural Network - LSTM

BATCH_SIZE = 5000
BUFFER_SIZE = 100000

# Shuffle, batch, and cache the dataset
train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.take(BATCH_SIZE).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).cache().repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.take(BATCH_SIZE).batch(BATCH_SIZE).cache().repeat()

# Create the LSTM model
lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
    tf.keras.layers.Dense(1)
])

# Compile the Model
lstm_model.compile(optimizer='adam', loss='mae')

for x, y in val_univariate.take(1):
    print(lstm_model.predict(x).shape)

# Train the Model
epochs = 10
steps_per_epoch = 200

trained_lstm = lstm_model.fit(
    train_univariate,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_univariate,
    validation_steps=50)

# Plot LSMT Model's 3 predictions
for x, y in val_univariate.take(1):
    plot = show_plot([x[0].numpy(), y[0].numpy(),
                      lstm_model.predict(x)[0]], 0, 'LSTM model')
    plot.show()

# Plot loss
loss = trained_lstm.history['loss']
val_loss = trained_lstm.history['val_loss']
epochs_range = range(epochs)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

