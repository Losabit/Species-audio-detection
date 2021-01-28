import os
import csv
import matplotlib.pyplot as plt
import soundfile as sf
import math
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
number_augmented_data_per_extract = 2


def save_spectrogramm(data, sample, picture_path):
    xx, frequency, bins, im = plt.specgram(data, Fs=sample)
    plt.axis('off')
    plt.savefig(picture_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    image = Image.open(picture_path)
    image.convert('RGB').resize((IMAGE_WIDTH, IMAGE_HEIGHT)).save(picture_path)


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
    total_lines = count_csv_lines(metadata_inpath)
    one_percent = int(total_lines / 100)
    percent = 0
    with open(metadata_inpath, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count % one_percent == 0:
                if percent % PERCENT_PRINT == 0:
                    print(str(percent) + "%")
                percent += 1

            if TRAIN_PERCENT > line_count / total_lines:
                class_directory = os.path.join(DATASET_TRAIN_DIRECTORY, str(row["species_id"]))
                if not os.path.isdir(class_directory):
                    os.mkdir(class_directory)
            else:
                class_directory = os.path.join(DATASET_VAL_DIRECTORY, str(row["species_id"]))
                if not os.path.isdir(class_directory):
                    os.mkdir(class_directory)

            input_path = os.path.join(audio_inpath, row["recording_id"] + ".flac")
            output_path = os.path.join(class_directory, row["recording_id"] + "_" + str(line_count))
            data, sample = sf.read(input_path)

            if DATASET_TRAIN_DIRECTORY in output_path:

                # element 0 ne compte pas; il faut donc faire +1 pour avoir le bon nombre d'element augmenter
                for j in range(number_augmented_data_per_extract + 1):

                    extension = ".png"
                    if j != 0:
                        data = augmentations[j - 1](samples=data, sample_rate=sample)
                        extension = F"__{j}.png"

                    process_and_save_spectrogramm(data,
                                                  sample,
                                                  output_path,
                                                  extension,
                                                  float(row["t_min"]), float(row["t_max"]))
            else:
                process_and_save_spectrogramm(data,
                                              sample,
                                              output_path,
                                              ".png",
                                              float(row["t_min"]), float(row["t_max"]))
            line_count += 1
        print('100%')
