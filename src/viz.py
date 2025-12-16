import numpy as np
import tensorflow as tf
import cv2


def grad_cam(model, img, class_idx, layer_name="last_convolution"):
    # Garante que o modelo foi inicializado
    model(img, training=False)

    conv_layer = model.get_layer(layer_name)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.outputs[0]]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img, training=False)

        # IMPORTANTE: dizer ao tape para observar conv_out
        tape.watch(conv_out)

        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)

    # Segurança extra
    if grads is None:
        return np.zeros((32, 32))

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    pooled_grads = pooled_grads.numpy()

    heatmap = np.zeros(conv_out.shape[:2], dtype=np.float32)

    for i in range(pooled_grads.shape[-1]):
        heatmap += pooled_grads[i] * conv_out[:, :, i]

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    return heatmap


def overlay_heatmap(img, heatmap):
    heatmap = cv2.resize(heatmap, (32, 32))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img = np.uint8(255 * img)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return overlay


