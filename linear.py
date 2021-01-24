import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.python.keras.layers import RandomFlip, RandomRotation, Dense, Flatten
from tensorflow.keras.optimizers import SGD
from dataset_param import *
import numpy as np
from utils import *


def create_base_model(add_custom_layers_func) -> Model:
    m = Sequential()
    add_custom_layers_func(m)

    m.add(Flatten())
    m.add(tf.keras.layers.Dense(len_classes, tf.keras.activations.softmax))
    m.add(Dense(len_classes, activation=tf.keras.activations.softmax))
    m.compile(optimizer=tf.keras.optimizers.SGD(),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=tf.keras.metrics.categorical_accuracy)

    return m


def linear_mod(Seq):
    pass


def train_model(m: Model, x, y, x_val, y_val):
    m.fit(
        x,
        y,
        validation_data=(x_val, y_val),
        epochs=epch,
        batch_size=1024
    )


if __name__ == '__main__':
    (data, label) = load_data()
    train_data, val_data = split_array(data, 0.8)
    train_labels, val_labels = split_array(label, 0.8)
    train_data, train_labels = build_x_y(train_data, train_labels)
    val_data, val_labels = build_x_y(val_data, val_labels)
    print("data loaded")

    model = create_base_model(linear_mod)
    train_model(model, train_data, train_labels, val_data, val_labels)
