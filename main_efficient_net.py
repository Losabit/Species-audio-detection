import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from efficientnet.keras import EfficientNetB0 as EfficientNet
from dataset_param import *
import numpy as np
from utils import *
from submission import *

train_size = compute_train_images_count()
val_size = compute_val_images_count()


def get_callbacks():
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2,
                                  verbose=1, mode='auto', min_delta=0.0001,
                                  cooldown=0, min_lr=0)

    early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=1)

    return [early_stopping, reduce_lr]


def create_efficient_net_models():
    inputs = layers.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))

    m = Sequential([
        EfficientNet(include_top=False, weights='imagenet', input_tensor=inputs),
        layers.GlobalAveragePooling2D(name="avg_pool"),
        layers.BatchNormalization(),
        layers.Dropout(0.2, name="top_dropout"),
        layers.Dense(len_classes, activation="softmax", name="pred")
    ])
    m.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.0001),
              metrics=['categorical_accuracy'])
    return m


def train_model(m, x_iterator, y_iterator):
    print(F"Training model...")
    log = m.fit(x_iterator,
                validation_data=y_iterator,
                steps_per_epoch=train_size // batch_size,
                validation_steps=val_size // batch_size,
                batch_size=batch_size, epochs=epch,
                callbacks=get_callbacks())
    return log


def create_dataset_iterator(base_folder: str, size: int):
    def inner_func():
        return tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(base_folder,
                                                                                                      target_size=(
                                                                                                          IMAGE_WIDTH,
                                                                                                          IMAGE_HEIGHT),
                                                                                                      batch_size=1)

    return (tf.data.Dataset.from_generator(inner_func,
                                           output_types=(tf.float32, tf.float32),
                                           output_shapes=(
                                               (1, *(IMAGE_WIDTH, IMAGE_HEIGHT), 3),
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
    model = create_efficient_net_models()

    all_logs = [

        {"value": train_model(model,
                              create_dataset_iterator(DATASET_TRAIN_DIRECTORY, train_size),
                              create_dataset_iterator(DATASET_VAL_DIRECTORY, val_size)),
         "title": F"efficient_net"}
    ]
    plot_all_logs(all_logs)

    print(F"Evaluation du model...")
    model.evaluate(create_dataset_iterator(DATASET_VAL_DIRECTORY, val_size),
                   steps=val_size // batch_size)

    print(F"Sauvegarde des prédictions a partir du modele...")
    predict_and_save_in_submission(model, average)
