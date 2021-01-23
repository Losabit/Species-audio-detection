import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import PIL
from PIL import Image


'''
https://stackoverflow.com/questions/23377665/python-scipy-fft-wav-files
https://docs.scipy.org/doc/scipy/reference/tutorial/fft.html
'''

inpath = os.path.join(os.getcwd(), 'dataset', 'rfcx-species-audio-detection')
metadata_inpath = os.path.join(inpath, 'train_tp.csv')
audio_inpath = os.path.join(inpath, 'train')

outpath = os.path.join(os.getcwd(), 'dataset', 'spectrogram-species-audio-detection')
metadata_outpath = os.path.join(outpath, 'train_tp.csv')
audio_outpath = os.path.join(outpath, 'train')

height = 32
width = 32
print_it = 200

if not os.path.isdir(outpath):
    os.mkdir(outpath)
if not os.path.isdir(audio_outpath):
    os.mkdir(audio_outpath)

with open(metadata_outpath, mode='w', newline='') as output_csv_file:
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

            data, sample = sf.read(os.path.join(audio_inpath, row["recording_id"] + ".flac"))
            row["recording_id"] = row["recording_id"] + "_" + str(line_count)
            picture = os.path.join(audio_outpath, row["recording_id"] + ".png")

            start_audio = int(float(row["t_min"]) * 48000)
            end_audio = int(float(row["t_max"]) * 48000)
            xx, frequency, bins, im = plt.specgram([data[i] for i in range(start_audio, end_audio)], Fs=sample)

            plt.axis('off')
            plt.savefig(picture, bbox_inches='tight', pad_inches=0)
            plt.close()

            image = Image.open(picture)
            image.resize((width, height)).save(picture)
            writer.writerow(row)
            line_count += 1
        print(f'Processed {line_count} lines.')
