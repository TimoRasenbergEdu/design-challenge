import os
import keras
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import gymnasium as gym
from keras import Input
from keras.models import Model


def jet_heatmap(heatmap, img_array, alpha):
    heatmap = np.uint8(255 * heatmap)

    jet = mpl.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + np.stack([img_array] * 3, axis=-1)
    return keras.utils.array_to_img(superimposed_img)


def gradcam_img(img_array, model, last_conv_layer_name, alpha=0.01):
    conv_model = Model(
        model.inputs,
        model.get_layer(last_conv_layer_name).output
    )

    pred_input = Input(
        shape=model.get_layer(last_conv_layer_name).output.shape[1:]
    )
    pred_model = pred_input
    add_layers = False
    for layer in model.layers:
        if add_layers:
            pred_model = layer(pred_model)

        if layer.name == last_conv_layer_name:
            add_layers = True

    pred_model = Model(pred_input, pred_model)

    with tf.GradientTape(watch_accessed_variables=False) as tape:
        last_conv_layer_output = conv_model(img_array[np.newaxis])
        tape.watch(last_conv_layer_output)
        preds = pred_model(last_conv_layer_output)
        class_channel = preds[:, tf.argmax(preds[0])]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return jet_heatmap(heatmap, img_array, alpha)


for dir in ['05-06-2024-20-12-24']:
    path = os.path.join('rl/history', dir)
    model = keras.models.load_model(os.path.join(path, 'model.keras'))

    model.summary()

    states = []

    env = gym.make("ALE/Qbert-v5", obs_type='grayscale')
    state, _ = env.reset()

    while True:
        action = env.action_space.sample()
        next_state, _, terminated, _, _ = env.step(action)

        state = state / 255.0
        states.append(state)

        state = next_state

        if terminated:
            break

    random_state_idx = np.random.choice(len(states), 10)
    for i in random_state_idx:
        img_array = states[i]
        gradcam = gradcam_img(img_array, model, 'conv2d_2')
        gradcam.save(f'gradcam_{i}.png')
