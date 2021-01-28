import os
import csv
import matplotlib.pyplot as plt
import soundfile as sf
import math
from PIL import Image
from dataset_param import *
from utils import count_csv_lines

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
                              sample,  output_path + "_" + str(it) + ".png")
            current_duration += duration
            it += 1

        if MINIMAL_DURATION < end_audio - current_duration - start_audio:
            # print("take rest : " + str(end_audio - current_duration - start_audio))
            save_spectrogramm([data[i] for i in range(int(start_audio + current_duration * initial_freq)
                                                      , int(end_audio * initial_freq))],
                              sample, output_path + "_r.png")

    elif MINIMAL_DURATION < end_audio - start_audio:
        save_spectrogramm([data[i] for i in range(int(start_audio * initial_freq), int(end_audio * initial_freq))],
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