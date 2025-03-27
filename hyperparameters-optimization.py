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

# Limpa o diretório do tuner
tuner_directory = 'my_dir'
if os.path.exists(tuner_directory):
    shutil.rmtree(tuner_directory)


# Selecionar 200 dados para treinamento de cada cenário
train_samples = 100
test_samples = 100


# Carregar os dados
PosSensor = 'VG'
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

# Normalização dos dados
def normalize_data(data, PosSensor='VG'):
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
        
        #Min-Max
        min_val = np.min(data)
        max_val = np.max(data)
        normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

# Aplicar normalização aos dados
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
window_size = 10
paa_size = int(5830/window_size)


# Função para criar o callback ReduceLROnPlateau
def build_reduce_lr_callback():
    return tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=20,
        min_lr=1e-6
    )

# Construção do Modelo 
def build_model(hp):
    
    model = tf.keras.Sequential()

    # Camada de entrada
    model.add(tf.keras.layers.Input(shape=(583, 1)))

    # Número total de camadas convolucionais (agora otimizado separadamente do pooling)
    num_conv_layers = hp.Int('num_conv_layers', min_value=2, max_value=5)

    for i in range(num_conv_layers):
        # Definição de hiperparâmetros específicos para cada camada convolucional
        filters = hp.Int(f'filters_{i}', min_value=32, max_value=128, step=16)
        kernel_size = hp.Int(f'kernel_size_{i}', min_value=2, max_value=5, step=1)

        # Adicionando a camada convolucional
        model.add(tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
        
        # Decisão se haverá MaxPooling após essa camada
        use_pooling = hp.Choice(f'use_pooling_{i}', values=[True, False])
        if use_pooling:
            model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

    # Dropout para regularização
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.3, step=0.1)
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))

    # Flatten
    model.add(tf.keras.layers.Flatten())

    # Camada densa oculta
    dense_units = hp.Int('dense_units', min_value=16, max_value=64, step=16)
    model.add(tf.keras.layers.Dense(units=dense_units, activation='relu'))

    # Camada de saída (10 classes)
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # Otimização da taxa de aprendizado
    learning_rate = hp.Choice('learning_rate', values=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

    return model

# Criação e configuração do tuner
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=100,
    executions_per_trial=3,
    directory='my_dir',
    project_name='cnn_tuning'
)

# Divisão dos dados
x_train = dadosRigidez.drop(['y'], axis=1).values
y_train = pd.get_dummies(dadosRigidez['y']).values
y_test = pd.get_dummies(dadosRigidezTeste['y']).values
x_test = dadosRigidezTeste.drop(['y'], axis=1).values

# Reshape para atender à entrada do modelo
x_train = x_train.reshape(-1, 583, 1)
x_test = x_test.reshape(-1, 583, 1)

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
    x_train, y_train, test_size=0.2, random_state=42)


# Treinamento do modelo
# Treinamento do modelo, incluindo batch_size como um hiperparâmetro
tuner.search(x_train, y_train,
             epochs=100,
             validation_data=(x_val, y_val),
             batch_size=tuner.oracle.hyperparameters.Int('batch_size', min_value=8, max_value=48, step=8),
             callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20),
                        build_reduce_lr_callback()])

# Exibir os melhores hiperparâmetros encontrados
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Melhores hiperparâmetros encontrados:")
for key, value in best_hyperparameters.values.items():
    print(f"{key}: {value}")

# Exibir o melhor modelo
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()
