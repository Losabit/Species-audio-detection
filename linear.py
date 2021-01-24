import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.python.keras.layers import RandomFlip, RandomRotation, Dense, Flatten
from tensorflow.keras.optimizers import SGD
from dataset_param import *
import  numpy as np
from utils import *


def create_base_model(add_custom_layers_func) -> Model:
    m = Sequential()
    add_custom_layers_func(m)

    m.add(Flatten())
    m.add(tf.keras.layers.Dense(10, tf.keras.activations.softmax))
    m.add(Dense(len(len_classes), activation=tf.keras.activations.softmax))
    m.compile(optimizer=tf.keras.optimizers.SGD(),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=tf.keras.metrics.categorical_accuracy)


    return m


def linearMod(Senq):
    pass


def train_models(m, x, y):
    m.fit(x, y, epochs=epch,batch_size=2)
    m.summary()


if __name__ == '__main__':
    m = create_base_model(linearMod)
    (data, label) = load_data()
    x , y  = build_x_y(data,label)
    train_models(m,x,y)
    print(x)
    print(m)
