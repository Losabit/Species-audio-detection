import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
#from wavio import readwav

'''
https://stackoverflow.com/questions/23377665/python-scipy-fft-wav-files
https://docs.scipy.org/doc/scipy/reference/tutorial/fft.html
'''

metadata_inpath = os.path.join(os.getcwd(), 'dataset/rfcx-species-audio-detection/train_tp.csv')
audio_inpath = os.path.join(os.getcwd(), 'dataset/rfcx-species-audio-detection/train')

outpath = os.path.join(os.getcwd(), 'dataset/spectrogram-species-audio-detection')
metadata_outpath = os.path.join(outpath, 'train_tp.csv')
audio_outpath = os.path.join(outpath, 'train')

if not os.path.isdir(outpath):
    os.mkdir(outpath)
if not os.path.isdir(audio_outpath):
    os.mkdir(audio_outpath)

with open(metadata_outpath, mode='w') as output_file:
    with open(metadata_inpath, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1

            if line_count == 1:
                data, sample = sf.read(os.path.join(audio_inpath, row["recording_id"] + ".flac"))
                xx, frequency, bins, im = plt.specgram(data, Fs=sample)
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [sec]')
                plt.show()

            line_count += 1
        print(f'Processed {line_count} lines.')