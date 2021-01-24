import os

DATASET_DIRECTORY = os.path.join(os.getcwd(), 'dataset', 'spectrogram-species-audio-detection')
DATASET_TRAIN_DIRECTORY = os.path.join(DATASET_DIRECTORY, 'train')
DATASET_TRUE_TRAIN_CSV = os.path.join(DATASET_DIRECTORY, 'train_tp.csv')
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
len_classes = 24
epch = 500
KERNEL_REGULARIZERS = 0.0005
lrTest =  0.01
momentumTest = 0.95
### PARAMS spectrogramm_conversion ###
# Lié à IMAGE_WIDTH et IMAGE_HEIGHT
print_it = 200
# duration_cut -> Découpage des extraits en morceaux de x secondes / 0 = pas de découpage
duration_cut = 2
# minimum duration of record
minimal_duration = 0.5
initial_freq = 48000
freq_modifier = 0
