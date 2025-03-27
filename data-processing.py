# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:39:29 2024

@author: Thiago Moreno Fernandes

Multi damage classification with Piecewise Aggregate Approximation (PAA) e CNN
"""


import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import KFold
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import time
import tkinter as tk

from scipy import stats
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import regularizers, layers, initializers
from tensorflow.keras.layers import Lambda
from tensorflow.keras.utils import to_categorical
from pyts.approximation import PiecewiseAggregateApproximation, SymbolicAggregateApproximation
from sklearn.preprocessing import RobustScaler, Normalizer
from scipy.stats import rankdata

        
plt.rcParams['font.family'] = 'Times New Roman' # Fonte dos plots

start_time = time.time()  # Record the start time

# Número de rodadas
n_runs = 10


# Selecionar 200 dados para treinamento de cada cenário
train_samples = 100
test_samples = 100

# Carregar os dados
PosSensor = 'TF'
Vagao = 'PrimVag'
# Carregar os dados
Dados_Baseline = loadmat(f'Data19-10-24_{PosSensor}_{Vagao}_Case1_Cut.mat') 
Dados_Case2 = loadmat(f'Data19-10-24_{PosSensor}_{Vagao}_Case2_Cut.mat') 
Dados_Case3 = loadmat(f'Data19-10-24_{PosSensor}_{Vagao}_Case3_Cut.mat') 
Dados_Case4 = loadmat(f'Data19-10-24_{PosSensor}_{Vagao}_Case4_Cut.mat') 
Dados_Case5 = loadmat(f'Data19-10-24_{PosSensor}_{Vagao}_Case5_Cut.mat') 
Dados_Case6 = loadmat(f'Data19-10-24_{PosSensor}_{Vagao}_Case6_Cut.mat')
Dados_Case7 = loadmat(f'Data19-10-24_{PosSensor}_{Vagao}_Case7_Cut.mat') 
Dados_Case8 = loadmat(f'Data19-10-24_{PosSensor}_{Vagao}_Case8_Cut.mat') 
Dados_Case9 = loadmat(f'Data19-10-24_{PosSensor}_{Vagao}_Case9_Cut.mat') 
Dados_Case10 = loadmat(f'Data19-10-24_{PosSensor}_{Vagao}_Case10_Cut.mat') 

dataBaseline = Dados_Baseline['Baseline']               
dataCase2 = Dados_Case2['Case2']                        
dataCase3 = Dados_Case3['Case3']                      
dataCase4 = Dados_Case4['Case4']                       
dataCase5 = Dados_Case5['Case5']                        
dataCase6 = Dados_Case6['Case6']                        
dataCase7 = Dados_Case7['Case7']                        
dataCase8 = Dados_Case8['Case8']                        
dataCase9 = Dados_Case9['Case9']                        
dataCase10 = Dados_Case10['Case10']  


# Normalização dos dados  Testar outras técnicas de normalização

## Normalização min-max
def normalize_data(data, PosSensor='TF'):
    if PosSensor == 'TF':
        # Min-Max normalization
        min_val = np.min(data)
        max_val = np.max(data)
        normalized_data = (data - min_val) / (max_val - min_val)
        
    else:
        # VSS normalization
        # mean_val = np.mean(data)
        # std_dev = np.std(data)
        # normalized_data = ((data - mean_val) / std_dev) * (mean_val / std_dev)  
        min_val = np.min(data)
        max_val = np.max(data)
        normalized_data = (data - min_val) / (max_val - min_val)
        
    return normalized_data

#Aplicar normalização aos dados
dataBaseline = normalize_data(dataBaseline)
dataCase2 = normalize_data(dataCase2)
dataCase3 = normalize_data(dataCase3)
dataCase4 = normalize_data(dataCase4)
dataCase5 = normalize_data(dataCase5)
dataCase6 = normalize_data(dataCase6)
dataCase7 = normalize_data(dataCase7)
dataCase8 = normalize_data(dataCase8)
dataCase9 = normalize_data(dataCase9)
dataCase10 = normalize_data(dataCase10)


dadosRigidez = pd.DataFrame()

# Amostragem dos dados
dataBaseline_df = pd.DataFrame(dataBaseline)
dataBaseline_df['y'] = 0
dataBaseline_sampled_train = dataBaseline_df.sample(n=train_samples, random_state=42)
dataBaseline_sampled_test = dataBaseline_df.drop(dataBaseline_sampled_train.index).sample(n=test_samples, random_state=42)
dadosRigidez = pd.concat([dadosRigidez, dataBaseline_sampled_train], ignore_index=True)

# Adicionar dataCase2 com coluna y_Case2
dataCase2_df = pd.DataFrame(dataCase2)
dataCase2_df['y'] = 1
dataCase2_sampled_train = dataCase2_df.sample(n=train_samples, random_state=42)
dataCase2_sampled_test = dataCase2_df.drop(dataCase2_sampled_train.index).sample(n=test_samples, random_state=42)
dadosRigidez = pd.concat([dadosRigidez, dataCase2_sampled_train], ignore_index=True)

# Adicionar dataDez com coluna y_Case3
dataCase3_df = pd.DataFrame(dataCase3)
dataCase3_df['y'] = 2
dataCase3_sampled_train = dataCase3_df.sample(n=train_samples, random_state=42)
dataCase3_sampled_test = dataCase3_df.drop(dataCase3_sampled_train.index).sample(n=test_samples, random_state=42)
dadosRigidez = pd.concat([dadosRigidez, dataCase3_sampled_train], ignore_index=True)

# Adicionar dataVinte com coluna y_Case4
dataCase4_df = pd.DataFrame(dataCase4)
dataCase4_df['y'] = 3
dataCase4_sampled_train = dataCase4_df.sample(n=train_samples, random_state=42)
dataCase4_sampled_test = dataCase4_df.drop(dataCase4_sampled_train.index).sample(n=test_samples, random_state=42)
dadosRigidez = pd.concat([dadosRigidez, dataCase4_sampled_train], ignore_index=True)

# Adicionar dataVinte com coluna y_Case5
dataCase5_df = pd.DataFrame(dataCase5)
dataCase5_df['y'] = 4
dataCase5_sampled_train = dataCase5_df.sample(n=train_samples, random_state=42)
dataCase5_sampled_test = dataCase5_df.drop(dataCase5_sampled_train.index).sample(n=test_samples, random_state=42)
dadosRigidez = pd.concat([dadosRigidez, dataCase5_sampled_train], ignore_index=True)

# Adicionar dataVinte com coluna y_Case6
dataCase6_df = pd.DataFrame(dataCase6)
dataCase6_df['y'] = 5
dataCase6_sampled_train = dataCase6_df.sample(n=train_samples, random_state=42)
dataCase6_sampled_test = dataCase6_df.drop(dataCase6_sampled_train.index).sample(n=test_samples, random_state=42)
dadosRigidez = pd.concat([dadosRigidez, dataCase6_sampled_train], ignore_index=True)

# Adicionar dataVinte com coluna y_Case6
dataCase7_df = pd.DataFrame(dataCase7)
dataCase7_df['y'] = 6
dataCase7_sampled_train = dataCase7_df.sample(n=train_samples, random_state=42)
dataCase7_sampled_test = dataCase7_df.drop(dataCase7_sampled_train.index).sample(n=test_samples, random_state=42)
dadosRigidez = pd.concat([dadosRigidez, dataCase7_sampled_train], ignore_index=True)

dataCase8_df = pd.DataFrame(dataCase8)
dataCase8_df['y'] = 7
dataCase8_sampled_train = dataCase8_df.sample(n=train_samples, random_state=42)
dataCase8_sampled_test = dataCase8_df.drop(dataCase8_sampled_train.index).sample(n=test_samples, random_state=42)
dadosRigidez = pd.concat([dadosRigidez, dataCase8_sampled_train], ignore_index=True)

dataCase9_df = pd.DataFrame(dataCase9)
dataCase9_df['y'] = 8
dataCase9_sampled_train = dataCase9_df.sample(n=train_samples, random_state=42)
dataCase9_sampled_test = dataCase9_df.drop(dataCase9_sampled_train.index).sample(n=test_samples, random_state=42)
dadosRigidez = pd.concat([dadosRigidez, dataCase9_sampled_train], ignore_index=True)

dataCase10_df = pd.DataFrame(dataCase10)
dataCase10_df['y'] = 9
dataCase10_sampled_train = dataCase10_df.sample(n=train_samples, random_state=42)
dataCase10_sampled_test = dataCase10_df.drop(dataCase10_sampled_train.index).sample(n=test_samples, random_state=42)
dadosRigidez = pd.concat([dadosRigidez, dataCase10_sampled_train], ignore_index=True)


dadosRigidezTeste = pd.concat([dataBaseline_sampled_test, dataCase2_sampled_test, dataCase3_sampled_test, dataCase4_sampled_test,
                                dataCase5_sampled_test, dataCase6_sampled_test, dataCase7_sampled_test, 
                                dataCase8_sampled_test, dataCase9_sampled_test, dataCase10_sampled_test], ignore_index=True)


    
# Preencher valores nulos, se houver
dadosRigidez = dadosRigidez.fillna(0)
dadosRigidezTeste = dadosRigidezTeste.fillna(0)

# Função para aplicar PAA nos dados
def apply_paa(data, paa_size):
    n_samples, n_features = data.shape
    step = n_features // paa_size
    return np.mean(data.reshape(n_samples, paa_size, step), axis=2)

# Aplicar PAA nos dados de treino e teste
window_size = 1
paa_size = int(5830/window_size)


def create_model(n_classes=10):
    
    #initializer = tf.keras.initializers.HeNormal()
    global PosSensor

    if PosSensor == 'TF':
        
        # Arquitetura para 'TF'
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

        

    elif PosSensor == 'VG':
        
        
        input_layer = tf.keras.layers.Input(shape=(paa_size, 1))
    
        cnn = tf.keras.layers.Conv1D(filters=48, kernel_size=2, activation='relu')(input_layer)
        cnn = tf.keras.layers.Conv1D(filters=48, kernel_size=3, activation='relu')(cnn)
        cnn = tf.keras.layers.Flatten()(cnn)
        dense = tf.keras.layers.Dense(48, activation='relu')(cnn)
        
        # Camada de saída
        output_layer = tf.keras.layers.Dense(10, activation='softmax')(dense)
        
        # Criar o modelo
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        
        # Compilar o modelo
        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      metrics=['accuracy'])

    

    return model



# Função para treinar o modelo e obter a matriz de confusão
def train_and_evaluate_confusion_matrix(model, x_train, y_train, x_test, y_test,i):
    best_accuracy = 0
    best_conf_matrix = None

    
    # Treinamento do modelo
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30, min_lr=1e-7, verbose=1, restore_best_weights=True)

    history = model.fit(x_train, y_train, epochs=1000, batch_size=24, validation_split=0.2, verbose=1, callbacks=[early_stopping, reduce_lr])

    # Plotando o decaimento da função custo
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch', fontsize=22)
    plt.ylabel('Loss', fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.savefig(f'Loss_{PosSensor}_WithoutPAA_{i}_Normalization_BayesOpt.jpg', dpi=300, bbox_inches='tight')     
    plt.show()

    # Predições
    ytestpred = model.predict(x_test)
    ytestpred_classes = np.argmax(ytestpred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    # Calculando a matriz de confusão e a precisão
    conf_matrix = confusion_matrix(y_test_classes, ytestpred_classes)
    accuracy = accuracy_score(y_test_classes, ytestpred_classes)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_conf_matrix = conf_matrix

    return best_conf_matrix, accuracy


        
# Inicializar variáveis para salvar a melhor matriz de confusão
best_overall_accuracy = 0
best_overall_conf_matrix = None
accuracies = {scenario: [] for scenario in ['Healthy','DC2', 'DC3','DC4', 'DC5', 'DC6', 'DC7', 'DC8', 'DC9', 'DC10']}


# Loop principal
for i in range(n_runs):
    print(f"Execução {i+1}/{n_runs}")
        


    # Dividir os dados em treinamento e teste
    x_train = dadosRigidez.drop(columns=['y']).values
    y_train = dadosRigidez['y'].values
    x_test = dadosRigidezTeste.drop(columns=['y']).values
    y_test = dadosRigidezTeste['y'].values
    
    # Transformar as labels em categorias (one-hot encoding)
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    
    # Convertendo para arrays NumPy com tipo float
    x_train = apply_paa(x_train, paa_size)
    x_test = apply_paa(x_test, paa_size)
    
    # Debug: Inspecione os shapes e tipos
    print("Shape de x_train:", x_train.shape, "Tipo:", x_train.dtype)
    print("Shape de x_test:", x_test.shape, "Tipo:", x_test.dtype)
    
    # Ajuste do reshape
    x_train = x_train.reshape(x_train.shape[0], paa_size, 1)
    x_test = x_test.reshape(x_test.shape[0], paa_size, 1)

    
    # x_train = x_train.astype(np.float32)
    # x_test = x_test.astype(np.float32)
    

    # Criar o modelo
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=i
    )

    # Criar o modelo para cada rodada
    model = create_model(n_classes=10)

    # Treinar o modelo e obter a melhor matriz de confusão
    best_conf_matrix, accuracy = train_and_evaluate_confusion_matrix(
        model, x_train, y_train, x_test, y_test, i
    )

    # Comparar acurácia para obter a melhor matriz de confusão
    if accuracy > best_overall_accuracy:
        best_overall_accuracy = accuracy
        best_overall_conf_matrix = best_conf_matrix

    # Calcular as acurácias para cada cenário individualmente
    ytestpred = model.predict(x_test)
    ytestpred_classes = np.argmax(ytestpred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    for j, scenario in enumerate(['Healthy','DC2', 'DC3','DC4', 'DC5', 'DC6', 'DC7', 'DC8', 'DC9', 'DC10']):
        scenario_mask = (y_test_classes == j)
        scenario_accuracy = accuracy_score(y_test_classes[scenario_mask], ytestpred_classes[scenario_mask])
        accuracies[scenario].append(scenario_accuracy)


#Plots
class_names = ['Healthy','DC2', 'DC3','DC4', 'DC5', 'DC6', 'DC7', 'DC8', 'DC9', 'DC10']

# Plotar matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(best_overall_conf_matrix, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 20}, xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted', fontsize=22)
plt.ylabel('True', fontsize=22)
#plt.title('Confusion Matrix', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig(f'ConfusionMatrix_{PosSensor}_Normalization_WithoutPAA_BayesOpt.png', dpi=300, bbox_inches='tight')
plt.show()

# Plotar boxplot das acurácias
plt.figure(figsize=(8, 6))
accuracies_df = pd.DataFrame(accuracies)

box = plt.boxplot([accuracies_df[col] for col in accuracies_df.columns],
                   patch_artist=True)  # Permite a personalização das caixas

# Personalização do boxplot
for i, patch in enumerate(box['boxes']):
    # Contornos das caixas sem preenchimento
    patch.set_edgecolor('black')  # Define a cor da borda
    #patch.set_facecolor('blue')  # Remove o preenchimento da caixa
    patch.set_linewidth(1.5)     # Espessura da borda

# Personalizar linhas medianas
for median in box['medians']:
    median.set_color('Orange')     # Cor da linha mediana
    median.set_linewidth(2)     # Espessura da linha mediana

# Personalizar whiskers e caps
for whisker in box['whiskers']:
    whisker.set_color('black')  # Cor dos whiskers
    whisker.set_linewidth(1.0)  # Espessura dos whiskers
    whisker.set_linestyle((0, (7, 5)))

for cap in box['caps']:
    cap.set_color('black')      # Cor dos caps
    cap.set_linewidth(1.0)      # Espessura dos caps

# Configurações adicionais do gráfico
plt.xlabel('Scenario', fontsize=24, fontfamily='serif', fontname='Times New Roman')
plt.ylabel('Accuracy', fontsize=24, fontfamily='serif', fontname='Times New Roman')
plt.xticks(ticks=range(1, len(accuracies_df.columns) + 1),
           labels=class_names, fontsize=20, fontfamily='serif', fontname='Times New Roman', rotation=45)
plt.yticks(fontsize=20, fontfamily='serif', fontname='Times New Roman')
plt.ylim(0, 1)  # Limitar eixo Y de 0 a 1
plt.grid(True, linestyle='--', alpha=0.7)


# Salvar a figura
plt.savefig(f'Boxplot_accuracy_{PosSensor}_Normalization_WithoutPAA_BayesOpt.png', dpi=300, bbox_inches='tight')

# Mostrar o gráfico
plt.show()

print("--- Tempo total de execução: %.2f minutos ---" % ((time.time() - start_time) / 60))