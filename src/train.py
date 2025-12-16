import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from data import load_data
from models import modelo_mlp, modelo_cnn
import csv
import os

# Possibilita argumentos via linha de comando
parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["mlp", "cnn"], required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--kernel", type=int, default=3)
args = parser.parse_args()

# Define a seed para reprodutibilidade (Usarei: 41, 42, 43)
tf.random.set_seed(args.seed)

# Carrega os dados
x_train, y_train, x_val, y_val, x_test, y_test = load_data()

#Define o nome do modelo
if args.model == "cnn":
    model_path = f"models/cnn_k{args.kernel}_seed{args.seed}.keras"
else:
    model_path = f"models/mlp_seed{args.seed}.h5"

# Usa Early Stopping para evitar overfitting
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True),
        
ModelCheckpoint(
        filepath=model_path,
        monitor="val_loss",
        save_best_only=True)
]

# Trata da escolha do modelo
if args.model == "mlp":
    model = modelo_mlp(input_shape=x_train.shape[1:])

elif args.model == "cnn":
    model = modelo_cnn(
        input_shape=x_train.shape[1:],
        kernel_size=(args.kernel, args.kernel)
    )

# Treino do modelo
history = model.fit(
    x_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# Avaliação do modelo no conjunto de teste
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Seed: {args.seed} | Modelo: {args.model} | Acc teste: {test_acc:.4f}")

# Salva resultados em CSV
os.makedirs("results", exist_ok=True)

result_file = "results/results.csv"
file_exists = os.path.isfile(result_file)

with open(result_file, mode="a", newline="") as f:
    writer = csv.writer(f)

    if not file_exists:
        writer.writerow([
            "model", "kernel", "seed",
            "test_accuracy"
        ])

    writer.writerow([
        args.model,
        args.kernel if args.model == "cnn" else "NA",
        args.seed,
        test_acc
    ])




    
