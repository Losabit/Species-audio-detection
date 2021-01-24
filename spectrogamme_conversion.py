import os
import csv
import matplotlib.pyplot as plt
import soundfile as sf
import math
from PIL import Image
import numpy as np
from dataset_param import *

'''
https://stackoverflow.com/questions/23377665/python-scipy-fft-wav-files
https://docs.scipy.org/doc/scipy/reference/tutorial/fft.html
'''
initial_freq = 48000
inpath = os.path.join(os.getcwd(), 'dataset', 'rfcx-species-audio-detection')
metadata_inpath = os.path.join(inpath, 'train_tp.csv')
audio_inpath = os.path.join(inpath, 'train')


def save_spectrogramm(data, sample, picture_path):
    xx, frequency, bins, im = plt.specgram(data, Fs=sample)
    plt.axis('off')
    plt.savefig(picture_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    image = Image.open(picture_path)
    image.resize((IMAGE_WIDTH, IMAGE_HEIGHT)).save(picture_path)


def process_and_save_spectrogramm(input_path, output_path, start_audio, end_audio):
    data, sample = sf.read(input_path)
    if duration_cut != 0:
        nb_extraits = (end_audio - start_audio) / duration_cut
        nb_extraits_int = int(math.floor(nb_extraits))
        for i in range(nb_extraits_int):
            save_spectrogramm([data[j] for j in range(int((start_audio + i * duration_cut) * initial_freq),
                                                      int((start_audio + (i + 1) * duration_cut) * initial_freq))],
                              sample, output_path + "_" + str(i) + ".png")

        if minimal_duration < end_audio - (nb_extraits_int * duration_cut) - start_audio:
            save_spectrogramm([data[i] for i in range(int((start_audio + nb_extraits_int * duration_cut) * initial_freq)
                                                      , int(end_audio * initial_freq))],
                              sample, output_path + "_" + str(nb_extraits_int) + ".png")

    elif minimal_duration < end_audio - start_audio:
        save_spectrogramm([data[i] for i in range(int(start_audio * initial_freq), int(end_audio * initial_freq))],
                          sample, output_path + "_0" + ".png")


def count_line(path):
    with open(metadata_inpath, mode='r') as file:
        reader = csv.DictReader(file)
        count = 0
        for _ in reader:
            count += 1
        return count


def create_spectro_dataset():
    
    total_lines = count_line(DATASET_TRUE_TRAIN_CSV)
    with open(metadata_inpath, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count % print_it == 0:
                print(str(line_count) + " extraits traités")

            if train_percent > line_count / total_lines:
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
        print(f'Processed {line_count} lines.')
