import matplotlib.pyplot as plt
import soundfile as sf
import math
from PIL import Image
from dataset_param import *

test_path = os.path.join(os.getcwd(), 'dataset', 'rfcx-species-audio-detection', 'test')
initial_freq = 48000


def save_spectrogramm(d, s, picture_path):
    xx, frequency, bins, im = plt.specgram(d, Fs=s)
    plt.axis('off')
    plt.savefig(picture_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    image = Image.open(picture_path)
    image.resize((IMAGE_WIDTH, IMAGE_HEIGHT)).save(picture_path)


if not os.path.isdir(DATASET_DIRECTORY):
    os.mkdir(DATASET_DIRECTORY)
if not os.path.isdir(DATASET_TEST_DIRECTORY):
    os.mkdir(DATASET_TEST_DIRECTORY)

line_count = 0
for file in os.listdir(test_path):
    file_path = (os.path.join(test_path, file))
    data, sample = sf.read(file_path)
    end_audio = len(data) / sample

    directory_music = os.path.join(DATASET_TEST_DIRECTORY, file.replace(".flac", ""))
    if not os.path.isdir(directory_music):
        os.mkdir(directory_music)
    new_file_path = (os.path.join(directory_music, file.replace(".flac", "")))

    if line_count % print_it == 0:
        print(str(line_count) + " extraits traités")

    if test_duration_cut != 0:
        nb_extraits = end_audio / test_duration_cut
        nb_extraits_int = int(math.floor(nb_extraits))
        for i in range(nb_extraits_int):
            save_spectrogramm([data[j] for j in range(int((i * test_duration_cut) * initial_freq),
                                                      int(((i + 1) * test_duration_cut) * initial_freq))],
                              sample,  new_file_path + "_" + str(i) + ".png")

        if test_minimal_duration < end_audio - (nb_extraits_int * duration_cut):
            save_spectrogramm([data[i] for i in range(int((nb_extraits_int * duration_cut) * initial_freq)
                                                      , int(end_audio * initial_freq))],
                              sample, new_file_path + "_" + str(nb_extraits_int) + ".png")
    else:
        save_spectrogramm(data, sample, new_file_path + ".png")

    line_count += 1

print("Extraction terminée")
