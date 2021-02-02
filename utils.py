from datetime import datetime
from dataset_param import *
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import image
from dataset_param import *
from PIL import Image


def save_spectrogramm(d, s, duration, picture_path):
    xx, frequency, bins, im = plt.specgram(d, Fs=s)
    plt.axis('off')
    plt.savefig(picture_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    image = Image.open(picture_path)
    img_width = int((duration / 10) * IMAGE_WIDTH)
    image.convert('RGB').resize((img_width, IMAGE_HEIGHT)).save(picture_path)


def load_data(path):
    labels = np.zeros(0, dtype=np.float32)
    data = np.zeros((0, IMAGE_HEIGHT, IMAGE_WIDTH, 4), dtype=np.float32)
    for _, directories, _ in os.walk(path):
        for directory in directories:
            directory_path = os.path.join(path, directory)
            for file in os.listdir(directory_path):
                labels = np.append(labels, int(directory))
                spectro_image = image.imread(os.path.join(directory_path, file))
                spectro_image = np.expand_dims(spectro_image, axis=0)
                data = np.concatenate((data, spectro_image), axis=0)
    return data, labels


def split_array(data_to_split, percent):
    if percent > 1:
        raise Exception("percent parameter need to be between 0 and 1")

    percent_indice = int(len(data_to_split) * percent)
    return np.array([data_to_split[i] for i in range(percent_indice)]), \
           np.array([data_to_split[i] for i in range(percent_indice, len(data_to_split))])


def build_x_y(x, y):
    return x, tf.keras.utils.to_categorical(y, len_classes)


def count_csv_lines(path):
    with open(path, mode='r') as file:
        reader = csv.DictReader(file)
        count = 0
        for _ in reader:
            count += 1
        return count


def plot_all_logs(logs):
    print(logs)
    metrics = ['loss', 'val_loss', 'categorical_accuracy', 'val_categorical_accuracy']
    for metric in metrics:
        for log in logs:
            y_coords = log['value'].history[metric]
            x_coords = list(range(len(y_coords)))
            plt.plot(x_coords, y_coords)
            plt.title(log['title'] + " - " + datetime.now().strftime("%Hh:%Mm:%Ss") + " - " + metric)
            plt.show()
