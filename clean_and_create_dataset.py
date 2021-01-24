import os
import shutil
import csv
from random import random

from dataset_param import *
from spectrogamme_conversion import *


def clean_dataset():
    for c in destination_classes:
        folder = f'{DATASET_TRAIN_DIRECTORY}/{c}'
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                os.remove(f'{folder}/{filename}')

        folder = f'{DATASET_VAL_DIRECTORY}/{c}'
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                os.remove(f'{folder}/{filename}')


def clean_and_create_dataset():

    if not os.path.isdir(DATASET_DIRECTORY):
        os.mkdir(DATASET_DIRECTORY)
    if not os.path.isdir(DATASET_TRAIN_DIRECTORY):
        os.mkdir(DATASET_TRAIN_DIRECTORY)
    if not os.path.isdir(DATASET_VAL_DIRECTORY):
        os.mkdir(DATASET_VAL_DIRECTORY)

    clean_dataset()

    create_spectro_dataset()


if __name__ == "__main__":
    clean_and_create_dataset()
