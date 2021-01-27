import os
import csv
import matplotlib.pyplot as plt
import soundfile as sf
import math
from audiomentations import Compose, AddGaussianNoise, AddGaussianSNR, FrequencyMask, PitchShift
from PIL import Image
from dataset_param import *
from utils import count_csv_lines

initial_freq = 48000
inpath = os.path.join(os.getcwd(), 'dataset', 'rfcx-species-audio-detection')
metadata_inpath = os.path.join(inpath, 'train_tp.csv')
audio_inpath = os.path.join(inpath, 'train')
number_augmented_data_per_extract = 3


def save_spectrogramm(data, sample, picture_path):
    xx, frequency, bins, im = plt.specgram(data, Fs=sample)
    plt.axis('off')
    plt.savefig(picture_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    image = Image.open(picture_path)
    image.convert('RGB').resize((IMAGE_WIDTH, IMAGE_HEIGHT)).save(picture_path)


def augment_data_and_save(data, sample, picture_path):
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        AddGaussianSNR(),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        FrequencyMask()
    ])

    save_spectrogramm(data, sample, picture_path)

    for i in range(number_augmented_data_per_extract):
        augmented_samples = augment(samples=data, sample_rate=sample)
        save_spectrogramm(augmented_samples, sample, picture_path.replace(".png", F"__{i}.png"))


def process_and_save_spectrogramm(input_path, output_path, start_audio, end_audio):
    data, sample = sf.read(input_path)
    if DURATION_CUT != 0:
        nb_extraits = (end_audio - start_audio) / DURATION_CUT
        nb_extraits_int = int(math.floor(nb_extraits))
        for i in range(nb_extraits_int):
            augment_data_and_save([data[j] for j in range(int((start_audio + i * DURATION_CUT) * initial_freq),
                                                          int((start_audio + (i + 1) * DURATION_CUT) * initial_freq))],
                                  sample, output_path + "_" + str(i) + ".png")

        if MINIMAL_DURATION < end_audio - (nb_extraits_int * DURATION_CUT) - start_audio:
            augment_data_and_save(
                [data[i] for i in range(int((start_audio + nb_extraits_int * DURATION_CUT) * initial_freq)
                                        , int(end_audio * initial_freq))],
                sample, output_path + "_" + str(nb_extraits_int) + ".png")

    elif MINIMAL_DURATION < end_audio - start_audio:
        augment_data_and_save([data[i] for i in range(int(start_audio * initial_freq), int(end_audio * initial_freq))],
                              sample, output_path + "_0" + ".png")


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

            process_and_save_spectrogramm(os.path.join(audio_inpath, row["recording_id"] + ".flac"),
                                          os.path.join(class_directory, row["recording_id"] + "_" + str(line_count)),
                                          float(row["t_min"]), float(row["t_max"]))
            line_count += 1
        print('100%')
