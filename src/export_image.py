import numpy as np
from PIL import Image
from data import load_data
import os

# Cria pasta para salvar imagens
os.makedirs("images", exist_ok=True)

# Carrega os dados
_, _, _, _, x_test, y_test = load_data()

# Escolha a classe desejada (ex: 0 = avião)
target_class = 0

# Encontra a primeira imagem dessa classe
idx = np.where(np.argmax(y_test, axis=1) == target_class)[0][0]

img = x_test[idx]

# Converte para uint8
img_uint8 = (img * 255).astype("uint8")

# Salva a imagem
Image.fromarray(img_uint8).save("images/aviao_test.png")

print("Imagem salva em images/aviao_test.png")
''