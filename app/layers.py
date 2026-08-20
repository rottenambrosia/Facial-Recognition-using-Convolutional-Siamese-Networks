# I see a line of cars and they're all painted black
# With flowers and my love, both never to come back
# I see people turn their heads and quickly look away
# Like a newborn baby, it just happens every day

# custom L1 distance layer

import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, Dense, Flatten, Input, Layer, MaxPooling2D
from keras.models import Model


class L1_distance (Layer) :
    def __init__ (self, **kwargs) :
        super().__init__()

    def call (self, input_embedding, validation_embedding) :
        return tf.math.abs(input_embedding - validation_embedding)

embedding = Sequential([
    Input((100, 100, 3), name = "input_layer"),
    Conv2D(64, (10, 10), activation = "relu", name = "conv_1"),
    MaxPooling2D(64, (2, 2), name = "max_pool_1", padding = "same"),
    Conv2D(128, (7, 7), activation = "relu", name = "conv_2"),
    MaxPooling2D(64, (2, 2), name = "max_pool_2", padding = "same"),
    Conv2D(128, (4, 4), activation = "relu", name = "conv_3"),
    MaxPooling2D(64, (2, 2), name = "max_pool_3", padding = "same"),
    Conv2D(256, (4, 4), activation = "relu", name = "conv_4"),
    Flatten(),
    Dense(4096, activation="sigmoid")
], name = "embedding_model")

class siamese (Layer) :
    def siamese () :
        input_image = Input(shape = (100, 100, 3), name = "input_image")
        validation_image = Input(shape = (100, 100, 3), name = "validation_image")
        siamese_layer = L1_distance()
        siamese_layer.__name__ = "distance"
        distances = siamese_layer(embedding(input_image), embedding(validation_image))
        classifier = Dense(1, activation = "sigmoid") (distances)
        return Model(inputs = [input_image, validation_image], outputs = classifier, name = "siamese_net")