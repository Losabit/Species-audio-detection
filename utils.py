import csv
import numpy as np
from matplotlib import image
from dataset_param import *


def load_data():
    labels = np.zeros(0, dtype=np.float32)
    data = np.zeros((0, IMAGE_HEIGHT, IMAGE_WIDTH, 4), dtype=np.float32)
    with open(DATASET_TRUE_TRAIN_CSV, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            labels = np.append(labels, row["species_id"])
            spectro_image = image.imread(os.path.join(DATASET_TRAIN_DIRECTORY, row["recording_id"] + ".png"))
            spectro_image = np.expand_dims(spectro_image, axis=0)
            data = np.concatenate((data, spectro_image), axis=0)
            line_count += 1
    return data, labels


def split_array(data_to_split, percent):
    if percent > 1:
        raise Exception("percent parameter need to be between 0 and 1")

    percent_indice = int(len(data_to_split) * percent)
    #ne pas utiliser cela
    return data_to_split[0..percent_indice]