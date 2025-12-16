import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Cria a função de trabalho de dados
def load_data():
    # Divide o conjunto de dados do CIFAR-10 em treino e teste
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Seleciona da classe 0 a 4 pra fazer o subconjunto pedido
    classes = [0, 1, 2, 3, 4]

    # Remove as outras classes do conjunto
    mask_train = np.isin(y_train, classes).flatten()
    mask_test = np.isin(y_test, classes).flatten()

    x_train = x_train[mask_train]
    y_train = y_train[mask_train]
    x_test = x_test[mask_test]
    y_test = y_test[mask_test]

    # Divide o treino em duas partes: treino e validção
    val_size = int(0.1 * len(x_train))

    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]

    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]

    # Preprocessa os dados
    y_train = to_categorical(y_train, 5)
    y_val = to_categorical(y_val, 5)
    y_test = to_categorical(y_test, 5)

    x_train = x_train.astype("float32") / 255
    x_val = x_val.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Retorna os dados preparados para uso
    return x_train, y_train, x_val, y_val, x_test, y_test