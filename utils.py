from dataset_param import *
import os
import csv
import numpy as np
import tensorflow as tf
from matplotlib import image
from dataset_param import *


def load_data():
    labels = np.zeros(0, dtype=np.float32)
    data = np.zeros((0, IMAGE_HEIGHT, IMAGE_WIDTH, 4), dtype=np.float32)
    with open(DATASET_TRUE_TRAIN_CSV, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            labels = np.append(labels, int(row["species_id"]))
            spectro_image = image.imread(os.path.join(DATASET_TRAIN_DIRECTORY, row["recording_id"] + ".png"))
            spectro_image = np.expand_dims(spectro_image, axis=0)
            data = np.concatenate((data, spectro_image), axis=0)
            line_count += 1
    return data, labels


def build_x_y(x, y):
    x_ = np.zeros((0, 32, 32, 3), dtype=np.float32)
    y_ = np.zeros(0, dtype=np.float32)
    print(y_)
    for i in range(1,len(y)):
        y_ = np.concatenate((y_, y[i]), axis=0)
    y_ = tf.keras.utils.to_categorical(y, 24)
    x_ = x_/255
    return x_, y_
