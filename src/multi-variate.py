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
df = read_csv(csv_directory)

# Extract the three variables
features_considered = ['Global_active_power', 'Global_reactive_power', 'Sub_metering_2']
features = df[features_considered]
features.index = df['datetime']

print(features.head())

train_split = 1500000

# Normilice the data
dataset = features.values
tf.keras.utils.normalize(dataset)


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


# Single step model
# In a single step setup, the model learns to predict a single point in
# the future based on some history provided.

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])

    return np.array(data), np.array(labels)


# 1440 observations per day, I observe 5 days in the past
past_history = 7200

# make 1 observation every 10 minutes, 6 per hour
step = 6

# predict the Global_reactive_power 12 hr in the future
future_target = 72

x_train_single, y_train_single = multivariate_data(dataset,
                                                   dataset[:, 1],
                                                   0,
                                                   train_split,
                                                   past_history,
                                                   future_target,
                                                   step,
                                                   single_step=True)

x_val_single, y_val_single = multivariate_data(dataset,
                                               dataset[:, 1],
                                               train_split,
                                               None,
                                               past_history,
                                               future_target,
                                               step,
                                               single_step=True)

print('Single window of past history : {}'.format(x_train_single[0].shape))

batch_size = 5000
buffer_size = 100000

train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
train_data_single = train_data_single.take(batch_size).shuffle(buffer_size).batch(batch_size).cache().repeat()

val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.take(batch_size).batch(batch_size).shuffle(buffer_size).repeat()

single_step_model = tf.keras.models.Sequential()
single_step_model.add(tf.keras.layers.LSTM(32, input_shape=x_train_single.shape[-2:]))
single_step_model.add(tf.keras.layers.Dense(1))

single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

for x, y in val_data_single.take(1):
    print(single_step_model.predict(x).shape)

epochs = 10
steps_per_epoch = 200

single_step_history = single_step_model.fit(train_data_single,
                                            epochs=epochs,
                                            steps_per_epoch=steps_per_epoch,
                                            validation_data=val_data_single,
                                            validation_steps=50)


def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()


plot_train_history(single_step_history,
                   'Single Step Training and validation loss')

for x, y in val_data_single.take(1):
    plot = show_plot([x[0][:, 1].numpy(), y[0].numpy(),
                      single_step_model.predict(x)[0]], 12,
                     'Single Step Prediction')
    plot.show()

# Multi Step Model
# A multi-step model predicts a sequence of the future.

future_target = 72

x_train_multi, y_train_multi = multivariate_data(dataset,
                                                 dataset[:, 1],
                                                 0,
                                                 train_split,
                                                 past_history,
                                                 future_target,
                                                 step)

x_val_multi, y_val_multi = multivariate_data(dataset,
                                             dataset[:, 1],
                                             train_split,
                                             None,
                                             past_history,
                                             future_target,
                                             step)

print('Single window of past history : {}'.format(x_train_multi[0].shape))
print('\n Target Global_Reactive_Power to predict : {}'.format(y_train_multi[0].shape))

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.take(batch_size).shuffle(buffer_size).batch(batch_size).cache().repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.take(batch_size).batch(batch_size).cache().repeat()


def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 1]), label='History')
    plt.plot(np.arange(num_out) / step, np.array(true_future), 'bo',
             label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out) / step, np.array(prediction), 'ro',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()


for x, y in train_data_multi.take(1):
    multi_step_plot(x[0], y[0], np.array([0]))

multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32, return_sequences=True, input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
multi_step_model.add(tf.keras.layers.Dense(72))

multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

# Let's see how the model predicts before it trains.
for x, y in val_data_multi.take(1):
    print(multi_step_model.predict(x).shape)

multi_step_history = multi_step_model.fit(train_data_multi,
                                          epochs=epochs,
                                          steps_per_epoch=steps_per_epoch,
                                          validation_data=val_data_multi,
                                          validation_steps=50)

plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')

for x, y in val_data_multi.take(3):
    multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])
