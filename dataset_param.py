import os

DATASET_DIRECTORY = os.path.join(os.getcwd(), 'dataset', 'spectrogram-species-audio-detection')
DATASET_TRAIN_DIRECTORY = os.path.join(DATASET_DIRECTORY, 'train')
DATASET_VAL_DIRECTORY = os.path.join(DATASET_DIRECTORY, 'val')
DATASET_TEST_DIRECTORY = os.path.join(DATASET_DIRECTORY, 'test')
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
len_classes = 24
epch = 100
KERNEL_REGULARIZERS = 0.0005
ref_lr = 0.03
ref_batch_size = 1024
batch_size = 256
momentumTest = 0.95
destination_classes = [str(i) for i in range(len_classes)]

### PARAMS spectrogramm_conversion ###
# Lié à IMAGE_WIDTH et IMAGE_HEIGHT
PERCENT_PRINT = 10
# duration_cut -> Découpage des extraits en morceaux de x secondes / 0 = pas de découpage
DURATION_CUT = 2
# minimum duration of record
MINIMAL_DURATION = 0.5
FREQ_MODIFIER = 0
TRAIN_PERCENT = 0.8

### PARAMS test_spectrogramm_conversion ###
TEST_DURATION_CUT = 5
TEST_MINIMAL_DURATION = 1


def compute_class_images_count(base_folder: str, class_name: str):
    return sum((1 for _ in os.listdir(f'{base_folder}/{class_name}')))


def compute_all_classes_images_count(base_folder: str):
    return sum((compute_class_images_count(base_folder, c) for c in destination_classes))


def compute_train_images_count():
    return compute_all_classes_images_count(DATASET_TRAIN_DIRECTORY)


def compute_val_images_count():
    return compute_all_classes_images_count(DATASET_VAL_DIRECTORY)

