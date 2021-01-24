from datetime import datetime

from dataset_param import *
import csv
import numpy as np
import matplotlib.pyplot as plt
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
            if line_count % print_it == 0:
                print(str(line_count) + " chargés en memoires")
            labels = np.append(labels, int(row["species_id"]))
            spectro_image = image.imread(os.path.join(DATASET_TRAIN_DIRECTORY, row["recording_id"] + ".png"))
            spectro_image = np.expand_dims(spectro_image, axis=0)
            data = np.concatenate((data, spectro_image), axis=0)
            line_count += 1
    return data, labels


def split_array(data_to_split, percent):
    if percent > 1:
        raise Exception("percent parameter need to be between 0 and 1")

    percent_indice = int(len(data_to_split) * percent)
    return np.array([data_to_split[i] for i in range(percent_indice)]),\
           np.array([data_to_split[i] for i in range(percent_indice, len(data_to_split))])


def build_x_y(x, y):
    return x, tf.keras.utils.to_categorical(y, len_classes)

def plot_all_logs(logs):
    metrics = ['loss', 'val_loss', 'categorical_accuracy', 'val_categorical_accuracy']
    for metric in metrics:
        for log in logs:
            y_coords = log['value'].history[metric]
            x_coords = list(range(len(y_coords)))
            plt.plot(x_coords, y_coords)
            plt.title(log['title'] + " - " + datetime.now().strftime("%Hh:%Mm:%Ss") + " - " + metric)
            plt.show()

