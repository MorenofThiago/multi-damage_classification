# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:39:29 2024

@author: Thiago Moreno Fernandes

Multi-damage classification with Piecewise Aggregate Approximation (PAA) and Convolutional Neural Network
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

        
plt.rcParams['font.family'] = 'Times New Roman' 

start_time = time.time()  # Record the start time

# Train and test samples
train_samples = 100
test_samples = 100

# Number of the algorithm executions
n_runs = 20

# Select the positioning of the sensor to be evaluated
# CB: Car body; FB: Front bogie
PosSensor = 'CB'  

cases = ['Baseline', 'Case2', 'Case3', 'Case4', 'Case5', 'Case6', 'Case7', 'Case8', 'Case9', 'Case10']

data = {}
for case in cases:
    data[case] = loadmat(f'Data_{PosSensor}_{case}.mat')[case]

## Min-Max normalization
def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

#Apply the normalization
for case in cases:
    data[case] = normalize_data(data[case])

# Data sampling
y_values = range(10)  # To associate 0 with 'Baseline', 1 with 'Case2', ..., 9 with 'Case10'

# Initialize the DataFrame for concatenating the data
dataConc = pd.DataFrame()

# Initialize the DataFrame for the test data
dataConcTeste = pd.DataFrame()

# Loop to iterate over the cases and process them
for case, y in zip(cases, y_values):
    # Load the data dynamically using the case name
    data_case = data[case]  
    
    # Create a DataFrame and add the 'y' column
    data_case_df = pd.DataFrame(data_case)
    data_case_df['y'] = y
    
    # Sampling for training data
    data_case_sampled_train = data_case_df.sample(n=train_samples, random_state=42)
    data_case_sampled_test = data_case_df.drop(data_case_sampled_train.index).sample(n=test_samples, random_state=42)
    
    # Concatenate the training data to dataConc
    dataConc = pd.concat([dataConc, data_case_sampled_train], ignore_index=True)
    
    # Concatenate the test data to dataConcTeste
    dataConcTeste = pd.concat([dataConcTeste, data_case_sampled_test], ignore_index=True)

    
# Fills any missing (NaN) values with 0 if there are any
dataConc = dataConc.fillna(0)
dataConcTeste = dataConcTeste.fillna(0)

# PAA function
def apply_paa(data, paa_size):
    n_samples, n_features = data.shape
    step = n_features // paa_size
    return np.mean(data.reshape(n_samples, paa_size, step), axis=2)

# Applies PAA to the data
window_size = 10
paa_size = int(5830/window_size)

#CNN model for CB dataset and for FB dataset
def create_model(n_classes=10):
    
    global PosSensor

    if PosSensor == 'FB':
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(128, 3, activation='relu',  strides=1, input_shape=(paa_size, 1)),
            tf.keras.layers.MaxPooling1D(2),  
            tf.keras.layers.Conv1D(96, 3, activation='relu',  strides=1),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(96, 3, activation='relu',  strides=1),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(96, activation='relu'),
            tf.keras.layers.Dense(n_classes, activation='softmax')])
        model.compile(loss='categorical_crossentropy', 
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
                      metrics=['accuracy'])
            
    elif PosSensor == 'CB':        
        input_layer = tf.keras.layers.Input(shape=(paa_size, 1))   
        cnn = tf.keras.layers.Conv1D(filters=48, kernel_size=2, activation='relu')(input_layer)
        cnn = tf.keras.layers.Conv1D(filters=48, kernel_size=3, activation='relu')(cnn)
        cnn = tf.keras.layers.Flatten()(cnn)
        dense = tf.keras.layers.Dense(48, activation='relu')(cnn)
        output_layer = tf.keras.layers.Dense(10, activation='softmax')(dense)
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      metrics=['accuracy'])
            
    return model

# Function for training the model and obtaining the confusion matrix
def train_and_evaluate_confusion_matrix(model, x_train, y_train, x_test, y_test,i):
    best_accuracy = 0
    best_conf_matrix = None

    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30, min_lr=1e-7, verbose=1, restore_best_weights=True)

    history = model.fit(x_train, y_train, epochs=1000, batch_size=24, validation_split=0.2, verbose=1, callbacks=[early_stopping, reduce_lr])

    # Function decay plot
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch', fontsize=22)
    plt.ylabel('Loss', fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.savefig(f'Loss_{PosSensor}_{i}.jpg', dpi=300, bbox_inches='tight')     
    plt.show()

    # Predictions
    ytestpred = model.predict(x_test)
    ytestpred_classes = np.argmax(ytestpred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test_classes, ytestpred_classes)
    accuracy = accuracy_score(y_test_classes, ytestpred_classes)
        
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_conf_matrix = conf_matrix

    return best_conf_matrix, accuracy

# Initialize variables to save the best confusion matrix
best_overall_accuracy = 0
best_overall_conf_matrix = None
accuracies = {scenario: [] for scenario in ['Healthy','DC2', 'DC3','DC4', 'DC5', 'DC6', 'DC7', 'DC8', 'DC9', 'DC10']}

# Main loop
for i in range(n_runs):
    print(f"Execução {i+1}/{n_runs}")
        
    # Split data into training and testing
    x_train = dataConc.drop(columns=['y']).values
    y_train = dataConc['y'].values
    x_test = dataConcTeste.drop(columns=['y']).values
    y_test = dataConcTeste['y'].values
    
    # One-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    # Convert to NumPy arrays with float type
    x_train = apply_paa(x_train, paa_size)
    x_test = apply_paa(x_test, paa_size)
    
    # Reshape adjustment
    x_train = x_train.reshape(x_train.shape[0], paa_size, 1)
    x_test = x_test.reshape(x_test.shape[0], paa_size, 1)

    # Divides the training data into two sets: training and validation
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=i
    )

    # Create the model for each execution
    model = create_model(n_classes=10)

    # Train the model and get the best confusion matrix
    best_conf_matrix, accuracy = train_and_evaluate_confusion_matrix(
        model, x_train, y_train, x_test, y_test, i
    )

    # Compare accuracy to get the best confusion matrix
    if accuracy > best_overall_accuracy:
        best_overall_accuracy = accuracy
        best_overall_conf_matrix = best_conf_matrix

    # Calculates accuracies for each scenario individually
    ytestpred = model.predict(x_test)
    ytestpred_classes = np.argmax(ytestpred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    for j, scenario in enumerate(['Healthy','DC2', 'DC3','DC4', 'DC5', 'DC6', 'DC7', 'DC8', 'DC9', 'DC10']):
        scenario_mask = (y_test_classes == j)
        scenario_accuracy = accuracy_score(y_test_classes[scenario_mask], ytestpred_classes[scenario_mask])
        accuracies[scenario].append(scenario_accuracy)

#Plots
class_names = ['Healthy','DC2', 'DC3','DC4', 'DC5', 'DC6', 'DC7', 'DC8', 'DC9', 'DC10']

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(best_overall_conf_matrix, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 20}, xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted', fontsize=22)
plt.ylabel('True', fontsize=22)
#plt.title('Confusion Matrix', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Save the best confusion matrix plot
plt.savefig(f'ConfusionMatrix_{PosSensor}.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot the boxplot
plt.figure(figsize=(8, 6))
accuracies_df = pd.DataFrame(accuracies)

box = plt.boxplot([accuracies_df[col] for col in accuracies_df.columns],
                   patch_artist=True) 

for i, patch in enumerate(box['boxes']):
    patch.set_edgecolor('black')  
    #patch.set_facecolor('blue')  
    patch.set_linewidth(1.5)     

for median in box['medians']:
    median.set_color('Orange')     
    median.set_linewidth(2)     

for whisker in box['whiskers']:
    whisker.set_color('black')  
    whisker.set_linewidth(1.0)  
    whisker.set_linestyle((0, (7, 5)))

for cap in box['caps']:
    cap.set_color('black')   
    cap.set_linewidth(1.0)    

plt.xlabel('Scenario', fontsize=24, fontfamily='serif', fontname='Times New Roman')
plt.ylabel('Accuracy', fontsize=24, fontfamily='serif', fontname='Times New Roman')
plt.xticks(ticks=range(1, len(accuracies_df.columns) + 1),
           labels=class_names, fontsize=20, fontfamily='serif', fontname='Times New Roman', rotation=45)
plt.yticks(fontsize=20, fontfamily='serif', fontname='Times New Roman')
plt.ylim(0, 1)  
plt.grid(True, linestyle='--', alpha=0.7)


# Salve the boxplot plot
plt.savefig(f'Boxplot_{PosSensor}.jpg', dpi=300, bbox_inches='tight')
plt.show()

print("--- Total execution time: %.2f minutes ---" % ((time.time() - start_time) / 60))
