import matplotlib.pyplot as plt
import soundfile as sf
import math
from dataset_param import *
from utils import save_spectrogramm

test_path = os.path.join(ORIGINAL_DATASET_DIRECTORY, 'test')
initial_freq = 48000


def create_test_spectro_dataset():
    one_percent = int(sum([len(files) for r, d, files in os.walk(test_path)]) / 100)
    percent = 0
    line_count = 0
    for file in os.listdir(test_path):
        file_path = (os.path.join(test_path, file))
        data, sample = sf.read(file_path)
        end_audio = len(data) / sample

        directory_music = os.path.join(DATASET_TEST_DIRECTORY, file.replace(".flac", ""))
        if not os.path.isdir(directory_music):
            os.mkdir(directory_music)
        new_file_path = (os.path.join(directory_music, file.replace(".flac", "")))

        if line_count % one_percent == 0:
            if percent % PERCENT_PRINT == 0:
                print(str(percent) + "%")
            percent += 1

        duration = DURATION_CUT
        # if RANDOM_CUT:
        #  duration += random.randint(0, len())

        current_duration = 0
        it = 0
        while current_duration <= end_audio:
            save_spectrogramm([data[j] for j in range(int(current_duration * initial_freq),
                              len(data) - 1 if len(data) <= (int((current_duration + duration) * initial_freq))
                              else (int((current_duration + duration) * initial_freq)))],
                              duration,
                              sample, new_file_path + "_" + str(it) + ".png")
            current_duration += duration
            it += 1

        if MINIMAL_DURATION < end_audio - current_duration:
            save_spectrogramm([data[i] for i in range(int(current_duration * initial_freq)
                                                      , int(end_audio * initial_freq))],
                              (end_audio - current_duration),
                              sample, new_file_path + "_r.png")

        line_count += 1

    print("100%")
