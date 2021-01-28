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

    duration = TEST_DURATION_CUT
    if TEST_RANDOM_CUT:
        duration = random.randint(0, int(end_audio))
    if duration != 0:
        current_duration = 0
        it = 0
        while current_duration <= end_audio:
            if TEST_RANDOM_CUT:
                duration = random.randint(1, int(end_audio) - current_duration + 1)
                if duration + current_duration > end_audio:
                    break
            elif current_duration + duration > end_audio:
                break
            save_spectrogramm([data[j] for j in range(int(current_duration * initial_freq),
                                                      int((current_duration + duration) * initial_freq))],
                              sample,  new_file_path + "_" + str(it) + ".png")
            current_duration += duration
            it += 1

        if TEST_MINIMAL_DURATION < end_audio - current_duration:
            save_spectrogramm([data[i] for i in range(int(current_duration * initial_freq)
                                                      , int(end_audio * initial_freq))],
                              sample, new_file_path + "_r.png")
    else:
        save_spectrogramm(data, sample, new_file_path + ".png")

    line_count += 1

print("100%")
