from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

# MODELO MLP

# Define a função do modelo MLP
def modelo_mlp(input_shape):
    # Cria o modelo MLP
    model_mlp = Sequential([
        # Adiciona os processamenos e camadas
        Flatten(input_shape=input_shape),

        Dense(128, activation='relu',
                kernel_regularizer=regularizers.l2(1e-4)),

        Dropout(0.5),
        
        Dense(5, activation='softmax',
                kernel_regularizer=regularizers.l2(1e-4))
        ])
    
        # Compila o modelo com o otimizador Adam
    model_mlp.compile(loss='categorical_crossentropy',
                        optimizer=Adam(learning_rate=0.0005),
                        metrics=['accuracy'])
    
    # Retorna o modelo compilado
    return model_mlp


# MODELO CNN

# Define a função do modelo CNN passando tamanho do kernel como parâmetro
def modelo_cnn(input_shape, kernel_size):
    # Cria o modelo CNN
    model_cnn = Sequential(name=f"cnn_k{kernel_size[0]}")

    #Adiciona o Input
    model_cnn.add(Input(shape=input_shape, name="input"))
    
    # Adiciona as camadas convolucionais e de pooling
    model_cnn.add(
        Conv2D(
            32,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
            kernel_regularizer=regularizers.l2(1e-4)))

    model_cnn.add(MaxPooling2D((2, 2)))

    model_cnn.add(Conv2D(
            64,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
            kernel_regularizer=regularizers.l2(1e-4)))

    model_cnn.add(MaxPooling2D((2, 2)))

    model_cnn.add(Conv2D(
            128,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
            kernel_regularizer=regularizers.l2(1e-4),
            name="last_convolution"))

    # Adiciona a camada de pooling global
    model_cnn.add(GlobalAveragePooling2D())

    #Utiliza a camada Dense para a classificação
    model_cnn.add(Dense(
            128,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)))

    model_cnn.add(Dropout(0.5))

    model_cnn.add(
        Dense(
            5,
            activation="softmax",
            kernel_regularizer=regularizers.l2(1e-4)))

    # Compila o modelo com o otimizador Adam
    model_cnn.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    # Retorna o modelo compilado
    return model_cnn