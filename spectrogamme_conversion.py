import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

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

if not os.path.isdir(outpath):
    os.mkdir(outpath)
if not os.path.isdir(audio_outpath):
    os.mkdir(audio_outpath)

with open(metadata_outpath, mode='w', newline='') as output_csv_file:
    with open(metadata_inpath, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                writer = csv.DictWriter(output_csv_file, row)
                writer.writeheader()
                line_count += 1

            data, sample = sf.read(os.path.join(audio_inpath, row["recording_id"] + ".flac"))
            start_audio = int(float(row["t_min"]) * 48000)
            end_audio = int(float(row["t_max"]) * 48000)
            xx, frequency, bins, im = plt.specgram([data[i] for i in range(start_audio, end_audio)], Fs=sample)
            plt.axis('off')
            row["recording_id"] = row["recording_id"] + "_" + str(line_count)
            plt.savefig(os.path.join(audio_outpath, row["recording_id"] + ".png"), bbox_inches='tight', pad_inches=0)
            plt.close()
            writer.writerow(row)
            line_count += 1
        print(f'Processed {line_count} lines.')
