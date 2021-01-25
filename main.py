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


def train_model(m: Model, x_iterator, y_iterator):
    log = m.fit(
        x_iterator,
        validation_data=y_iterator,
        steps_per_epoch=train_size // batch_size,
        validation_steps=val_size // batch_size,
        epochs=epch,
        class_weight=class_w
    )
    return log


def create_dataset_iterator(base_folder: str, size: int):
    def inner_func():
        return tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(base_folder,
                                                                                                      target_size=(
                                                                                                          IMAGE_WIDTH,
                                                                                                          IMAGE_HEIGHT),
                                                                                                      color_mode='rgba',
                                                                                                      batch_size=1)

    return (tf.data.Dataset.from_generator(inner_func,
                                           output_types=(tf.float32, tf.float32),
                                           output_shapes=(
                                               (1, *(IMAGE_WIDTH, IMAGE_HEIGHT), 4),
                                               (1, len_classes)
                                           )
                                           )
            .take(size)
            .unbatch()
            .batch(batch_size)
            .cache(f'{base_folder}/cache')
            .repeat()
            .as_numpy_iterator()
            )


if __name__ == '__main__':
    model = create_base_model(add_convnet)
    all_logs = [
        {"value": train_model(model,
                              create_dataset_iterator(DATASET_TRAIN_DIRECTORY, train_size),
                              create_dataset_iterator(DATASET_VAL_DIRECTORY, val_size)),
         "title": "add_convnet"}
    ]
    plot_all_logs(all_logs)

    print("Evaluation du dataset : ")
    print("[Train] => ")
    model.evaluate(create_dataset_iterator(DATASET_TRAIN_DIRECTORY, train_size),
                   steps=train_size // batch_size)

    print("[Validation] => ")
    model.evaluate(create_dataset_iterator(DATASET_VAL_DIRECTORY, val_size),
                   steps=val_size // batch_size)

    print("Sauvegarde des prédictions sur le jeu de test : ")
    predict_and_save_in_submission(model, average)


