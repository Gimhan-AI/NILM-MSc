import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Parameters
appliance_name = "kettle"
batch_size = 32
crop = 1000
network_type = "seq2point"
training_directory = "/media/vegaai/HardDrive/ProjectMSc/projectMSc-seq2point/dataset_management/refit/kettle/kettle_training_.csv"
validation_directory = "/media/vegaai/HardDrive/ProjectMSc/projectMSc-seq2point/dataset_management/refit/kettle/kettle_validation_H5.csv"
save_model_dir = "saved_models/" + appliance_name + "_" + network_type + "_model.h5"
epochs = 10
input_window_length = 599
validation_frequency = 1
patience = 3
min_delta = 1e-6
verbose = 1
loss = "mse"
metrics = ["mse", "msle", "mae"]
learning_rate = 0.001
beta_1 = 0.9
beta_2 = 0.999
shuffle = True

window_size = 2 + input_window_length
window_offset = int((0.5 * window_size) - 1)
offset = window_offset
max_chunk_size = 5 * 10 ** 2
ram_threshold = 5 * 10 ** 5
skip_rows_train = 10000000
validation_steps = 100
skip_rows_val = 0

# Data Loader Function
class TrainSlidingWindowGenerator:
    def __init__(self, file_name, chunk_size, shuffle, offset, batch_size=1000, crop=100000, skip_rows=0, ram_threshold=5 * 10 ** 5):
        self.file_name = file_name
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.offset = offset
        self.crop = crop
        self.skip_rows = skip_rows
        self.ram_threshold = ram_threshold
        self.total_size = 0
        self.total_num_samples = crop

    def load_dataset(self):
        print("Loading dataset from: ", self.file_name)
        data_array = np.array(pd.read_csv(self.file_name, nrows=self.crop, skiprows=self.skip_rows, header=0))
        inputs = data_array[:, 0]
        outputs = data_array[:, 1]
        maximum_batch_size = inputs.size - 2 * self.offset
        self.total_num_samples = maximum_batch_size
        if self.batch_size < 0:
            self.batch_size = maximum_batch_size

        indicies = np.arange(maximum_batch_size)
        if self.shuffle:
            np.random.shuffle(indicies)

        while True:
            for start_index in range(0, maximum_batch_size, self.batch_size):
                splice = indicies[start_index : start_index + self.batch_size]
                input_data = np.array([inputs[index : index + 2 * self.offset + 1] for index in splice])
                output_data = outputs[splice + self.offset].reshape(-1, 1)
                yield input_data, output_data

# Model Creation Function
def create_model(input_window_length):
    input_layer = tf.keras.layers.Input(shape=(input_window_length,))
    reshape_layer = tf.keras.layers.Reshape((1, input_window_length, 1))(input_layer)
    conv_layer_1 = tf.keras.layers.Convolution2D(filters=30, kernel_size=(10, 1), strides=(1, 1), padding="same", activation="relu")(reshape_layer)
    conv_layer_2 = tf.keras.layers.Convolution2D(filters=30, kernel_size=(8, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_1)
    conv_layer_3 = tf.keras.layers.Convolution2D(filters=40, kernel_size=(6, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_2)
    conv_layer_4 = tf.keras.layers.Convolution2D(filters=50, kernel_size=(5, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_3)
    conv_layer_5 = tf.keras.layers.Convolution2D(filters=50, kernel_size=(5, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_4)
    flatten_layer = tf.keras.layers.Flatten()(conv_layer_5)
    label_layer = tf.keras.layers.Dense(1024, activation="relu")(flatten_layer)
    output_layer = tf.keras.layers.Dense(1, activation="linear")(label_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model

# Load Dataset for Training
training_chunker = TrainSlidingWindowGenerator(file_name=training_directory, 
                                               chunk_size=max_chunk_size,
                                               batch_size=batch_size,
                                               crop=crop,
                                               shuffle=shuffle,
                                               skip_rows=skip_rows_train,
                                               offset=offset,
                                               ram_threshold=ram_threshold)

# Create and Compile the Model
model = create_model(input_window_length)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2), 
              loss=loss, 
              metrics=metrics)

# Set Early Stopping Callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=min_delta, patience=patience, verbose=verbose, mode="auto")
callbacks = [early_stopping]

# Calculate Steps per Epoch
steps_per_training_epoch = np.round(int(training_chunker.total_num_samples / batch_size), decimals=0) if training_chunker.total_num_samples is not None else 1

# Train the Model
training_history = model.fit(training_chunker.load_dataset(),
                             steps_per_epoch=steps_per_training_epoch,
                             epochs=epochs,
                             verbose=verbose,
                             callbacks=callbacks,
                             validation_steps=validation_steps,
                             validation_freq=validation_frequency)

# Save the Model
if not os.path.exists(save_model_dir):
    open((save_model_dir), 'a').close()
model.save(save_model_dir)

# Plot Training Results
plt.plot(training_history.history["loss"], label="MSE (Training Loss)")
if "val_loss" in training_history.history:
    plt.plot(training_history.history["val_loss"], label="MSE (Validation Loss)")
plt.title('Training History')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
