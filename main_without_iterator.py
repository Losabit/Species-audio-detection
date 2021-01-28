import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.python.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from dataset_param import *
import numpy as np
from utils import *
from submission import *

train_size = compute_train_images_count()
val_size = compute_val_images_count()
class_w = compute_class_weight()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


def create_base_model(add_custom_layers_func) -> Model:
    m = Sequential()
    add_custom_layers_func(m)

    m.add(Flatten())
    m.add(tf.keras.layers.Dense(len_classes, tf.keras.activations.softmax))

    m.compile(optimizer=tf.keras.optimizers.SGD(lr=ref_lr / ref_batch_size * batch_size),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=["categorical_accuracy"])

    return m


def linear_mod(Seq):
    pass


def add_mlp_layers(model):
    model.add(tf.keras.layers.Flatten())
    for _ in range(5):
        model.add(tf.keras.layers.Dense(2048, activation=tf.keras.activations.linear))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation(activation=tf.keras.activations.tanh))


def add_convnet(model):
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


def train_model(m: Model, x_train, y_train, x_val, y_val):
    log = m.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        steps_per_epoch=train_size // batch_size,
        validation_steps=val_size // batch_size,
        epochs=epch,
        class_weight=class_w
    )
    return log


if __name__ == '__main__':
    model = create_base_model(add_convnet)
    x_train, y_train = load_data(DATASET_TRAIN_DIRECTORY)
    x_val, y_val = load_data(DATASET_VAL_DIRECTORY)
    y_train = tf.keras.utils.to_categorical(y_train, len_classes)
    y_val = tf.keras.utils.to_categorical(y_val, len_classes)
    all_logs = [
        {"value": train_model(model,
                              x_train,
                              y_train,
                              x_val,
                              y_val),
         "title": "add_convnet"}
    ]
    plot_all_logs(all_logs)

    print("Sauvegarde des prédictions sur le jeu de test : ")
    predict_and_save_in_submission(model, average)
