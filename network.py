import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, Layer, MaxPooling2D
from tensorflow.keras.models import Model, Sequential


class L1Distance(Layer):
    """Compute the absolute difference between two embeddings."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


L1_distance = L1Distance


def build_embedding_network(input_shape=(100, 100, 3)):
    """Build the shared CNN used by the Siamese network."""
    embedding = Sequential(
        [
            Input(input_shape, name="input_layer"),
            Conv2D(64, (10, 10), activation="relu", name="conv_1"),
            MaxPooling2D((2, 2), name="max_pool_1", padding="same"),
            Conv2D(128, (7, 7), activation="relu", name="conv_2"),
            MaxPooling2D((2, 2), name="max_pool_2", padding="same"),
            Conv2D(128, (4, 4), activation="relu", name="conv_3"),
            MaxPooling2D((2, 2), name="max_pool_3", padding="same"),
            Conv2D(256, (4, 4), activation="relu", name="conv_4"),
            Flatten(),
            Dense(4096, activation="sigmoid"),
        ],
        name="embedding_model",
    )
    return embedding


embedding = build_embedding_network()
embedding_model = embedding


def build_siamese_network(input_shape=(100, 100, 3)):
    """Create a Siamese comparison network from the shared embedding model."""
    input_image = Input(shape=input_shape, name="input_image")
    validation_image = Input(shape=input_shape, name="validation_image")

    distance_layer = L1Distance(name="distance")
    distances = distance_layer(embedding(input_image), embedding(validation_image))
    classifier = Dense(1, activation="sigmoid")(distances)

    model = Model(
        inputs=[input_image, validation_image],
        outputs=classifier,
        name="siamese_net",
    )
    return model


def siamese():
    """Notebook-compatible wrapper returning the Siamese model."""
    return build_siamese_network()


siamese_neural_network = siamese()


if __name__ == "__main__":
    siamese_neural_network.summary()
