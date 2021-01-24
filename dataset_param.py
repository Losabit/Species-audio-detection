import os

DATASET_DIRECTORY = os.path.join(os.getcwd(), 'dataset', 'spectrogram-species-audio-detection')
DATASET_TRAIN_DIRECTORY = os.path.join(DATASET_DIRECTORY, 'train')
DATASET_VAL_DIRECTORY = os.path.join(DATASET_DIRECTORY, 'val')
DATASET_TEST_DIRECTORY = os.path.join(DATASET_DIRECTORY, 'test')
DATASET_TRUE_TRAIN_CSV = os.path.join(DATASET_DIRECTORY, 'train_tp.csv')
IMAGE_HEIGHT = 100
IMAGE_WIDTH = 100
len_classes = 24
epch = 100
TRAIN_PERCENT_DATA = 0.8
KERNEL_REGULARIZERS = 0.0005
ref_lr = 0.03
ref_batch_size = 1024
batch_size = 256
momentumTest = 0.95
destination_classes = [str(i) for i in range(len_classes)]
### PARAMS spectrogramm_conversion ###
# Lié à IMAGE_WIDTH et IMAGE_HEIGHT
print_it = 100
# duration_cut -> Découpage des extraits en morceaux de x secondes / 0 = pas de découpage
duration_cut = 2
# minimum duration of record
minimal_duration = 0.5
freq_modifier = 0
train_percent = 0.8

### PARAMS test_spectrogramm_conversion ###
test_duration_cut = 5
test_minimal_duration = 1


def compute_class_images_count(base_folder: str, class_name: str):
    return sum((1 for _ in os.listdir(f'{base_folder}/{class_name}')))


def compute_all_classes_images_count(base_folder: str):
    return sum((compute_class_images_count(base_folder, c) for c in destination_classes))


def compute_train_images_count():
    return compute_all_classes_images_count(DATASET_TRAIN_DIRECTORY)


def compute_val_images_count():
    return compute_all_classes_images_count(DATASET_VAL_DIRECTORY)
