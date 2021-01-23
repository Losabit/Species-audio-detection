import os
import csv
import numpy as np
from matplotlib import image

DATASET_DIRECTORY = os.path.join(os.getcwd(), 'dataset', 'spectrogram-species-audio-detection')
DATASET_TRAIN_DIRECTORY = os.path.join(DATASET_DIRECTORY, 'train')
DATASET_TRUE_TRAIN_CSV = os.path.join(DATASET_DIRECTORY, 'train_tp.csv')
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32

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


if __name__ == '__main__':
    data, labels = load_data()
    print(data.shape)
    print(labels.shape)