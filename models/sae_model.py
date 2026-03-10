import tensorflow as tf
from tensorflow.keras import layers

def build_sae(input_dim):

    input_layer = layers.Input(shape=(input_dim,))

    x = layers.Dense(64, activation="relu")(input_layer)
    x = layers.Dense(32, activation="relu")(x)

    latent = layers.Dense(16, activation="relu")(x)

    x = layers.Dense(32, activation="relu")(latent)
    x = layers.Dense(64, activation="relu")(x)

    output_layer = layers.Dense(input_dim, activation="sigmoid")(x)

    model = tf.keras.Model(input_layer, output_layer)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse"
    )

    return model
