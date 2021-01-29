import os
import csv
import matplotlib.pyplot as plt
import soundfile as sf
import pandas as pd
from audiomentations import Compose, AddGaussianNoise, AddGaussianSNR, FrequencyMask
from PIL import Image
from dataset_param import *
from utils import count_csv_lines

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
inpath = os.path.join(os.getcwd(), 'dataset', 'rfcx-species-audio-detection')
metadata_inpath = os.path.join(inpath, 'train_tp.csv')
audio_inpath = os.path.join(inpath, 'train')
number_augmented_data_per_extract = 0


def save_spectrogramm(data, sample, picture_path):
    xx, frequency, bins, im = plt.specgram(data, Fs=sample)
    plt.axis('off')
    plt.savefig(picture_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    image = Image.open(picture_path)
    image.convert('RGB').resize((IMAGE_WIDTH, IMAGE_HEIGHT)).save(picture_path)


def process_data_augmentation_and_save_spectrogramm(d, sample_rate, output_path, start, end):
    # element 0 ne compte pas; il faut donc faire +1 pour avoir le bon nombre d'element augmenter
    if end - start < MINIMAL_DURATION:
        return
    for j in range(number_augmented_data_per_extract + 1):
        extension = ".png"
        if j != 0:
            data = augmentations[j - 1](samples=d, sample_rate=sample_rate)
            extension = F"__{j}.png"

        process_and_save_spectrogramm(d,
                                      sample_rate,
                                      output_path,
                                      extension,
                                      start, end)


def process_and_save_spectrogramm(data, sample, output_path, extension, start_audio, end_audio):
    duration = DURATION_CUT

    if RANDOM_CUT:
        duration = random.randint(0, int(end_audio - start_audio))
    # print(duration)
    if duration != 0:
        current_duration = 0
        it = 0
        while current_duration <= end_audio - start_audio:
            if RANDOM_CUT:
                duration = random.randint(1, int(end_audio - start_audio) - current_duration + 1)
                if duration + current_duration + start_audio > end_audio:
                    break
            elif start_audio + current_duration + duration > end_audio:
                break
            # print("current : " + str(current_duration) + " / duration : " + str(duration))
            save_spectrogramm([data[j] for j in range(int((start_audio + current_duration) * initial_freq),
                                                      int((start_audio + current_duration + duration) * initial_freq))],
                              sample,  output_path + "_" + str(it) + extension)
            current_duration += duration
            it += 1

        if MINIMAL_DURATION < end_audio - current_duration - start_audio:
            # print("take rest : " + str(end_audio - current_duration - start_audio))
            save_spectrogramm([data[i] for i in range(int(start_audio + current_duration * initial_freq)
                                                      , int(end_audio * initial_freq))],
                              sample, output_path + "_r" + extension)
    elif MINIMAL_DURATION < end_audio - start_audio:
        save_spectrogramm(
            [data[i] for i in range(int(start_audio * initial_freq), int(end_audio * initial_freq))],
            sample, output_path + "_0" + extension)


def create_spectro_dataset():
    train = pd.read_csv(metadata_inpath).sort_values("recording_id")
    if USE_EMPTY_CLASS:
        duplicated_records = train[train.duplicated(subset=['recording_id'], keep=False)] \
            .filter(items=['recording_id', 't_min', 't_max'])
        duplicated_finish = []
        empty_class_directory = os.path.join(DATASET_TRAIN_DIRECTORY, str(len_classes - 1))
        if not os.path.isdir(empty_class_directory):
            os.mkdir(empty_class_directory)

    one_percent = int(len(train) / 100)
    percent = 0
    for index, row in train.iterrows():
        if index % one_percent == 0:
            if percent % PERCENT_PRINT == 0:
                print(str(percent) + "%")
            percent += 1

        class_directory = os.path.join(DATASET_TRAIN_DIRECTORY, str(row["species_id"]))
        if not os.path.isdir(class_directory):
            os.mkdir(class_directory)

        output_path = os.path.join(class_directory, row["recording_id"] + "_" + str(index))
        data, sample = sf.read(os.path.join(audio_inpath, row["recording_id"] + ".flac"))
        start = 0
        end = len(data) // sample
        if USE_EMPTY_CLASS and index % RATIO_EMPTY_CLASS == 0:
            print(index)
            output_empty_path = os.path.join(empty_class_directory, row["recording_id"] + "_" + str(index))
            if row['recording_id'] in duplicated_finish:
                continue
            elif len(duplicated_records[duplicated_records['recording_id'] == row['recording_id']]) > 0:
                records = duplicated_records[duplicated_records['recording_id'] == row['recording_id']].sort_values(
                    "t_min")
                duplicated_finish.append(row['recording_id'])
                it = 0
                process_data_augmentation_and_save_spectrogramm(data, sample, output_empty_path + "_" + str(start),
                                                                start, records.iloc[0]['t_min'])
                while it < len(records):
                    process_data_augmentation_and_save_spectrogramm(data, sample, output_path + "_" +
                                                                    str(records.iloc[it]['t_min']),
                                                                    records.iloc[it]['t_min'], records.iloc[it]['t_max'])
                    if it + 1 < len(records) and records.iloc[it]['t_max'] < records.iloc[it + 1]['t_min']:
                        process_data_augmentation_and_save_spectrogramm(data, sample, output_empty_path + "_" +
                                                                        str(records.iloc[it]['t_max']),
                                                                        records.iloc[it]['t_max'], records.iloc[it + 1]['t_min'])
                    it += 1
                process_data_augmentation_and_save_spectrogramm(data, sample, output_empty_path + "_" +
                                                                str(records.iloc[len(records) - 1]['t_max']),
                                                                records.iloc[len(records) - 1]['t_max'], end)
            else:
                process_data_augmentation_and_save_spectrogramm(data, sample, output_empty_path + "_" + str(1),
                                                                start, row['t_min'])
                process_data_augmentation_and_save_spectrogramm(data, sample, output_path, row['t_min'], row['t_max'])
                process_data_augmentation_and_save_spectrogramm(data, sample, output_empty_path + "_" + str(2),
                                                                row['t_max'], end)
        else:
            process_data_augmentation_and_save_spectrogramm(data, sample, output_path, row['t_min'], row['t_max'])

    print('100%')
