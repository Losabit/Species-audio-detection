import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.python.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from dataset_param import *
import numpy as np
from utils import *


def create_base_model(add_custom_layers_func) -> Model:
    m = Sequential()
    add_custom_layers_func(m)

    m.add(Flatten())
    m.add(tf.keras.layers.Dense(len_classes, tf.keras.activations.softmax))

    m.compile(optimizer=tf.keras.optimizers.SGD(lr=lrTest),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=["categorical_accuracy"])

    return m


def linear_mod(Seq):
    pass

def add_mlp_layers(model):
    model.add(tf.keras.layers.Flatten())
    for _ in range(5):
        model.add(tf.keras.layers.Dense(2048,activation=tf.keras.activations.linear))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation(activation=tf.keras.activations.tanh))

def add_convNet(model):
    model.add(tf.keras.layers.Reshape((IMAGE_WIDTH, IMAGE_HEIGHT, 4)))

    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.keras.activations.tanh,
                                     kernel_regularizer=tf.keras.regularizers.l2(KERNEL_REGULARIZERS)))
    model.add(tf.keras.layers.MaxPool2D())

    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.keras.activations.tanh,
                                     kernel_regularizer=tf.keras.regularizers.l2(KERNEL_REGULARIZERS)))
    model.add(tf.keras.layers.MaxPool2D())

    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.keras.activations.tanh,
                                     kernel_regularizer=tf.keras.regularizers.l2(KERNEL_REGULARIZERS)))
    model.add(tf.keras.layers.MaxPool2D())


def train_model(m: Model, x, y, x_val, y_val):
   log =  m.fit(
        x,
        y,
        validation_data=(x_val, y_val),
        epochs=epch,
        batch_size=batch_size
    )
   return log




if __name__ == '__main__':
    print("loading data")
    (data, label) = load_data()
    train_data, val_data = split_array(data, 0.8)
    train_labels, val_labels = split_array(label, 0.8)
    train_data, train_labels = build_x_y(train_data, train_labels)
    val_data, val_labels = build_x_y(val_data, val_labels)
    print("data loaded")

    model = create_base_model(add_convNet)
    all_logs = [
        {"value": train_model(model, train_data, train_labels, val_data, val_labels), "title": "Conv mod"},
    ]
    plot_all_logs(all_logs)
