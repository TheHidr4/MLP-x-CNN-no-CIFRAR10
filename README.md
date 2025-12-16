# Comparação CNN x MLP — CIFAR-10

Este projeto é um aplicativo em Streamlit para comparar modelos de Redes Neurais
treinados no dataset CIFAR-10.

O foco do projeto é:
- Comparar CNN com kernel 3×3 e 5×5
- Visualizar Grad-CAM
- Avaliar desempenho com matriz de confusão
- Fazer previsão de imagens enviadas pelo usuário



## Modelos utilizados

- CNN 3×3  
- CNN 5×5  
- MLP (para comparação)

Classes utilizadas (CIFAR-5):
- avião  
- automóvel  
- pássaro  
- gato  
- veado  



## Estrutura do projeto

MLP x CNN no CIFAR10/
- src/
  - app.py
  - viz.py
  - data.py
- models/
  - cnn_k3_seed42.keras
  - cnn_k5_seed41.keras
- venv/
- requirements.txt
- README.md



## Instalação

Criar ambiente virtual (opcional):

python -m venv venv  
venv\Scripts\activate  

Instalar dependências:

pip install -r requirements.txt



## Como executar

Entrar na pasta src:

cd src

Rodar o aplicativo:

streamlit run app.py

O navegador abrirá automaticamente.



## Funcionalidades

- Upload de imagem
- Previsão de classe
- Visualização Grad-CAM
- Comparação CNN 3×3 vs 5×5
- Matriz de confusão
- Comparação de desempenho entre MLP e CNN



## Bibliotecas usadas

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Matplotlib
- Pillow
- Scikit-learn
- cv2
- pandas
- argparse


## Objetivo

Projeto desenvolvido para estudo de Redes Neurais, CNNs, MLP e Visão Computacional.
