import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from scipy.io import loadmat
import random
from sklearn.model_selection import train_test_split
import keras_tuner as kt
import os
import shutil
from tensorflow.keras import regularizers, layers, initializers
from keras_tuner import HyperModel


# Clear the tuner directory
tuner_directory = 'my_dir'
if os.path.exists(tuner_directory):
    shutil.rmtree(tuner_directory)

# Select 200 data samples for training from each scenario
train_samples = 100
test_samples = 100

# Load the data
PosSensor = 'CB'

# Load and normalize the data
def load_and_normalize_data(pos_sensor, cases):
    data_dict = {}
    for i, case in enumerate(cases):
        data = loadmat(f'Data_{pos_sensor}_{case}.mat')[case]
        min_val, max_val = np.min(data), np.max(data)
        data_dict[case] = (data - min_val) / (max_val - min_val)  # Min-Max normalization
    return data_dict

# Initial definitions
cases = ['Baseline'] + [f'Case{i}' for i in range(2, 11)]
data_dict = load_and_normalize_data(PosSensor, cases)

dadosRigidez = pd.DataFrame()
dadosRigidezTeste = pd.DataFrame()

# Sampling and concatenation of the data
for i, case in enumerate(cases):
    df = pd.DataFrame(data_dict[case])
    df['y'] = i  # Assign corresponding label
    train_sampled = df.sample(n=train_samples, random_state=42)
    test_sampled = df.drop(train_sampled.index).sample(n=test_samples, random_state=42)
    
    dadosRigidez = pd.concat([dadosRigidez, train_sampled], ignore_index=True)
    dadosRigidezTeste = pd.concat([dadosRigidezTeste, test_sampled], ignore_index=True)

# Fill in null values, if any
dadosRigidez.fillna(0, inplace=True)
dadosRigidezTeste.fillna(0, inplace=True)

# Function to apply PAA to the data
def apply_paa(data, paa_size):
    n_samples, n_features = data.shape
    step = n_features // paa_size
    return np.mean(data.reshape(n_samples, paa_size, step), axis=2)

# Apply PAA to training and testing data
window_size = 10
paa_size = int(5830/window_size)

# Function to create the ReduceLROnPlateau callback
def build_reduce_lr_callback():
    return tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=20,
        min_lr=1e-6
    )

# Model construction
def build_model(hp):
    
    model = tf.keras.Sequential()

    # Input layer
    model.add(tf.keras.layers.Input(shape=(paa_size, 1)))

    # Total number of convolutional layers (now optimized separately from pooling)
    num_conv_layers = hp.Int('num_conv_layers', min_value=2, max_value=5)

    for i in range(num_conv_layers):
        # Define specific hyperparameters for each convolutional layer
        filters = hp.Int(f'filters_{i}', min_value=32, max_value=128, step=16)
        kernel_size = hp.Int(f'kernel_size_{i}', min_value=2, max_value=5, step=1)

        # Add the convolutional layer
        model.add(tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
        
        # Decide whether to apply MaxPooling after this layer
        use_pooling = hp.Choice(f'use_pooling_{i}', values=[True, False])
        if use_pooling:
            model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

    # Dropout for regularization
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.3, step=0.1)
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))

    # Flatten
    model.add(tf.keras.layers.Flatten())

    # Hidden dense layer
    dense_units = hp.Int('dense_units', min_value=16, max_value=64, step=16)
    model.add(tf.keras.layers.Dense(units=dense_units, activation='relu'))

    # Output layer (10 classes)
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # Learning rate optimization
    learning_rate = hp.Choice('learning_rate', values=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
    return model

# Create and configure the tuner
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=100,
    executions_per_trial=3,
    directory='my_dir',
    project_name='cnn_tuning'
)

# Data split
x_train = dadosRigidez.drop(['y'], axis=1).values
y_train = pd.get_dummies(dadosRigidez['y']).values
y_test = pd.get_dummies(dadosRigidezTeste['y']).values
x_test = dadosRigidezTeste.drop(['y'], axis=1).values

# Reshape to match the model input requirements
x_train = x_train.reshape(-1, 583, 1)
x_test = x_test.reshape(-1, 583, 1)

# Split the data into training and testing
x_train = dadosRigidez.drop(columns=['y']).values
y_train = dadosRigidez['y'].values
x_test = dadosRigidezTeste.drop(columns=['y']).values
y_test = dadosRigidezTeste['y'].values

# Transform labels into categorical (one-hot encoding)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


# Convert to NumPy arrays with float type
x_train = apply_paa(x_train, paa_size)
x_test = apply_paa(x_test, paa_size)

# Debug: Inspect shapes and types
print("Shape of x_train:", x_train.shape, "Type:", x_train.dtype)
print("Shape of x_test:", x_test.shape, "Type:", x_test.dtype)

# Adjust reshape
x_train = x_train.reshape(x_train.shape[0], paa_size, 1)
x_test = x_test.reshape(x_test.shape[0], paa_size, 1)

# Create the model
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42)

# Model training
# Train the model, including batch_size as a hyperparameter
tuner.search(x_train, y_train,
             epochs=100,
             validation_data=(x_val, y_val),
             batch_size=tuner.oracle.hyperparameters.Int('batch_size', min_value=8, max_value=48, step=8),
             callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20),
                        build_reduce_lr_callback()])

# Display the best hyperparameters found
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters found:")
for key, value in best_hyperparameters.values.items():
    print(f"{key}: {value}")

# Display the best model
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()
