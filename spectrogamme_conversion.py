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
    result = np.zeros((0, 2), dtype=np.float32)
    if duration_cut != 0:
        nb_extraits = (end_audio - start_audio) / duration_cut
        nb_extraits_int = int(math.floor(nb_extraits))
        for i in range(nb_extraits_int):
            save_spectrogramm([data[j] for j in range(int((start_audio + i * duration_cut) * initial_freq),
                                                      int((start_audio + (i + 1) * duration_cut) * initial_freq))],
                              sample,  output_path + "_" + str(i) + ".png")
            result = np.concatenate((result, [[start_audio + i * duration_cut,
                                               start_audio + (i + 1) * duration_cut]]), axis=0)

        if minimal_duration < end_audio - (nb_extraits_int * duration_cut) - start_audio:
            save_spectrogramm([data[i] for i in range(int((start_audio + nb_extraits_int * duration_cut) * initial_freq)
                                                      , int(end_audio * initial_freq))],
                              sample, output_path + "_" + str(nb_extraits_int) + ".png")
            result = np.concatenate((result, [[start_audio + nb_extraits_int * duration_cut,
                                               end_audio]]), axis=0)

    elif minimal_duration < end_audio - start_audio:
        save_spectrogramm([data[i] for i in range(int(start_audio * initial_freq), int(end_audio * initial_freq))],
                          sample, output_path + "_0" + ".png")
        result = np.concatenate((result, [[start_audio, end_audio]]), axis=0)
    return result


if not os.path.isdir(DATASET_DIRECTORY):
    os.mkdir(DATASET_DIRECTORY)
if not os.path.isdir(DATASET_TRAIN_DIRECTORY):
    os.mkdir(DATASET_TRAIN_DIRECTORY)

with open(DATASET_TRUE_TRAIN_CSV, mode='w', newline='') as output_csv_file:
    with open(metadata_inpath, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count % print_it == 0:
                print(str(line_count) + " extraits traités")

            if line_count == 0:
                writer = csv.DictWriter(output_csv_file, row)
                writer.writeheader()
                line_count += 1

            times_cut = process_and_save_spectrogramm(os.path.join(audio_inpath, row["recording_id"] + ".flac"),
                              os.path.join(DATASET_TRAIN_DIRECTORY, row["recording_id"] + "_" + str(line_count)),
                              float(row["t_min"]), float(row["t_max"]))

            recording_id_buff = row["recording_id"]
            for i in range(times_cut.shape[0]):
                row["recording_id"] = recording_id_buff + "_" + str(line_count) + "_" + str(i)
                row["t_min"] = str(times_cut[i][0])
                row["t_max"] = str(times_cut[i][1])
                writer.writerow(row)
            line_count += 1
        print(f'Processed {line_count} lines.')
