import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from models import modelo_cnn
from viz import grad_cam, overlay_heatmap


# Configura a página
st.set_page_config(
    page_title="Comparação CNN x MLP — CIFAR-10",
    layout="wide"
)

# Sidebar
view = st.sidebar.radio(
    "Selecione a visualização:",
    ["Grad-CAM: CNN 3×3 vs 5×5", "Desempenho: MLP vs CNN"])

# Grad-CAM view
if view == "Grad-CAM: CNN 3×3 vs 5×5":

    # Classes do CIFAR-5
    CLASSES = ["avião", "automóvel", "pássaro", "gato", "veado"]

    st.title("Comparação CNN k3 vs k5 – CIFAR-5")

    # Carrega modelos
    model_k3 = load_model("models/cnn_k3_seed42.keras")

    model_k5 = load_model("models/cnn_k5_seed41.keras")

    uploaded = st.file_uploader("Envie uma imagem (32x32 ou maior)", type=["png", "jpg", "jpeg"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img = img.resize((32, 32))
        st.image(img, caption="Imagem enviada", width=150)

        img_arr = np.array(img).astype("float32") / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        if st.button("Prever"):
            st.subheader("Resultados")

            for name, model in [("CNN k=3", model_k3), ("CNN k=5", model_k5)]:
                preds = model.predict(img_arr)[0]
                class_idx = np.argmax(preds)

                st.markdown(f"### {name}")

                for i, c in enumerate(CLASSES):
                    st.write(f"{c}: {preds[i]*100:.2f}%")

                heatmap = grad_cam(model, img_arr, class_idx)
                cam_img = overlay_heatmap(img_arr[0], heatmap)

                st.image(cam_img, caption="Grad-CAM")

                st.caption(
                    "O modelo focou nas regiões mais importantes da imagem para tomar a decisão."
            )


# Desempenho view
elif view == "Desempenho: MLP vs CNN":

    st.title("Comparação de Desempenho")
    df = pd.read_csv("results/escolhidos.csv")

    mlp_row = df[df["model"]=="mlp"].iloc[0]
    mlp_acc = mlp_row["test_accuracy"]
    mlp_epochs = int(mlp_row["epochs"])

    cnn_df = df[df["model"]=="cnn"]
    best_cnn_row = cnn_df.loc[cnn_df["test_accuracy"].idxmax()]
    best_cnn_acc = best_cnn_row["test_accuracy"]
    best_cnn_kernel = int(best_cnn_row["kernel"])
    best_cnn_epochs = int(best_cnn_row["epochs"])

    st.success(f"Melhor CNN: kernel {best_cnn_kernel}×{best_cnn_kernel}, "
               f"{best_cnn_epochs} épocas, acurácia {best_cnn_acc:.2%}")

    st.dataframe(df)

    fig, ax = plt.subplots()
    models = [f"MLP\n({mlp_epochs} épocas)", f"CNN {best_cnn_kernel}×{best_cnn_kernel}\n({best_cnn_epochs} épocas)"]
    accs = [mlp_acc, best_cnn_acc]
    ax.bar(models, accs)
    ax.set_ylim(0,1)
    ax.set_ylabel("Acurácia")
    ax.set_title("MLP vs Melhor CNN")
    st.pyplot(fig)