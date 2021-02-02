import os
import csv
import matplotlib.pyplot as plt
import soundfile as sf
import pandas as pd
from audiomentations import Compose, AddGaussianNoise, AddGaussianSNR, FrequencyMask
from dataset_param import *
from utils import count_csv_lines, save_spectrogramm

augmentations = [
    Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        AddGaussianSNR()
    ]),
    Compose([
        FrequencyMask()
    ])  # ,
    # TODO : AddBackgroundNoise
]

initial_freq = 48000
metadata_inpath = os.path.join(ORIGINAL_DATASET_DIRECTORY, 'train_tp.csv')
audio_inpath = os.path.join(ORIGINAL_DATASET_DIRECTORY, 'train')
number_extract_created = 0


def determine_class_directory(t_min, t_max, current_duration, duration, species_id, is_train):
    class_directory = ""
    dataset_directory = DATASET_TRAIN_DIRECTORY if is_train else DATASET_VAL_DIRECTORY

    if t_min <= current_duration <= t_max or \
            (t_min <= current_duration + duration <= t_max
             and t_max - (current_duration + duration) > MINIMAL_ANIMAL_PRESENCE) \
            or (current_duration <= t_min and current_duration + duration >= t_max):

        class_directory = os.path.join(dataset_directory, str(species_id))

    elif USE_EMPTY_CLASS and (current_duration + duration <= t_min or current_duration >= t_max):
        class_directory = os.path.join(dataset_directory, str(len_classes - 1))

    return class_directory


def process_data_and_save_spectrogramm(row_data, is_train):
    global number_extract_created
    current_duration = 0
    it = 0
    to_data_aug = 0
    t_min = row_data["t_min"]
    t_max = row_data["t_max"]
    recording_id = row_data["recording_id"]
    row_species = row_data["species_id"]

    data, sample = sf.read(os.path.join(audio_inpath, recording_id + ".flac"))
    end_audio = len(data) // sample

    while current_duration <= end_audio:

        duration = DURATION_CUT
        class_directory = ""
        create_empty_extract = False

        # if RANDOM_CUT:
        #  duration += random.randint(0, len())

        class_directory = determine_class_directory(t_min, t_max, current_duration, duration, row_species, is_train)
        is_empty_extract = str(len_classes - 1) in class_directory

        max_duration_size = len(data) - 1 if len(data) <= (int((current_duration + duration) * initial_freq)) \
            else (int((current_duration + duration) * initial_freq))

        if is_empty_extract and number_extract_created % RATIO_EMPTY_CLASS == 0:
            create_empty_extract = True

        if class_directory == "":
            current_duration += duration
            continue

        if is_empty_extract and not create_empty_extract:
            current_duration += duration
            continue

        extract_path = os.path.join(class_directory, recording_id)

        save_spectrogramm([data[j] for j in range(int(current_duration * initial_freq),
                                                  max_duration_size)],
                          sample,
                          duration,
                          extract_path + "_" + str(it) + ".png")

        if USE_DATA_AUGMENTATION and is_train is False and to_data_aug % RATIO_DATA_AUG == 0:
            new_data = augmentations[to_data_aug % 2](samples=data, sample_rate=sample)
            extract_path += F"_{str(it)}__{to_data_aug}.png"

            save_spectrogramm([new_data[j] for j in range(int(current_duration * initial_freq),
                                                          max_duration_size)],
                              sample,
                              duration,
                              extract_path)
            to_data_aug += 1

        current_duration += duration
        it += 1
        number_extract_created += 1

    if MINIMAL_DURATION < end_audio - current_duration:
        duration = DURATION_CUT
        row_species = row_data["species_id"]

        class_directory = determine_class_directory(t_min, t_max, current_duration, duration, row_species, is_train)
        extract_path = os.path.join(class_directory, recording_id)

        save_spectrogramm([data[i] for i in range(int(current_duration * initial_freq)
                                                  , int(end_audio * initial_freq))],
                          sample,
                          (end_audio - current_duration),
                          extract_path + "_r.png")
        number_extract_created += 1


def create_spectro_dataset():
    train = pd.read_csv(metadata_inpath).sort_values("recording_id")

    one_percent = int(len(train) / 100)
    percent = 0

    for index, row in train.iterrows():
        if index % one_percent == 0:
            if percent % PERCENT_PRINT == 0:
                print(str(percent) + "%")
            percent += 1

        if (1 - validation_split) > index / len(train):
            process_data_and_save_spectrogramm(row, is_train=True)
        else:
            process_data_and_save_spectrogramm(row, is_train=False)

    print('100%')
