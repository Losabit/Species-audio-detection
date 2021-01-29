import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from efficientnet.keras import EfficientNetB0 as EfficientNet
from dataset_param import *
import numpy as np
from utils import *
from submission import *
from mixup_data_generator import MixupImageDataGenerator


def get_callbacks():
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2,
                                  verbose=1, mode='auto', min_delta=0.0001,
                                  cooldown=0, min_lr=0)

    early_stopping = EarlyStopping(monitor="val_loss", patience=7, verbose=1)

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
    m.compile(loss=losses.CategoricalCrossentropy(),
              optimizer=optimizers.Adam(lr=0.0001),
              metrics=['categorical_accuracy'])
    return m


def train_model(m, x_iterator, y_iterator):
    log = m.fit(x_iterator,
                validation_data=y_iterator,
                steps_per_epoch=x_iterator.get_steps_per_epoch(),
                validation_steps=y_iterator.samples // batch_size,
                epochs=epch,
                callbacks=get_callbacks())
    return log


if __name__ == '__main__':
    print(F"Creating model...")
    model = create_efficient_net_models()

    input_imgen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

    train_generator = MixupImageDataGenerator(generator=input_imgen,
                                              directory=DATASET_TRAIN_DIRECTORY,
                                              batch_size=batch_size,  # To verify maybe error
                                              img_height=IMAGE_HEIGHT,
                                              img_width=IMAGE_WIDTH)

    validation_generator = input_imgen.flow_from_directory(directory=DATASET_VAL_DIRECTORY,
                                                           target_size=(
                                                               IMAGE_WIDTH,
                                                               IMAGE_HEIGHT),
                                                           batch_size=batch_size,  # To verify maybe error
                                                           class_mode="categorical",
                                                           shuffle=True)

    print('training steps: ', train_generator.get_steps_per_epoch())
    print('validation steps: ', validation_generator.samples // batch_size)

    print(F"Training model...")
    all_logs = [
        {"value": train_model(model,
                              train_generator,
                              validation_generator),
         "title": F"efficient_net"}
    ]
    plot_all_logs(all_logs)

    print(F"Evaluation du model...")

    validation_generator._set_index_array()
    model.evaluate(validation_generator,
                   steps=validation_generator.samples // batch_size)

    print(F"Sauvegarde des prédictions a partir du modele...")
    predict_and_save_in_submission(model, average)
