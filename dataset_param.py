import os, random
from pathlib import Path

PROJECT_PATH = os.getcwd()
if "Dataset_Creator" in PROJECT_PATH:
    PROJECT_PATH = Path(PROJECT_PATH).parent

ORIGINAL_DATASET_DIRECTORY = os.path.join(PROJECT_PATH, 'dataset', 'rfcx-species-audio-detection')
DATASET_DIRECTORY = os.path.join(PROJECT_PATH, 'dataset', 'spectrogram-species-audio-detection')
DATASET_TRAIN_DIRECTORY = os.path.join(DATASET_DIRECTORY, 'train')
DATASET_VAL_DIRECTORY = os.path.join(DATASET_DIRECTORY, 'val')
DATASET_TEST_DIRECTORY = os.path.join(DATASET_DIRECTORY, 'test')
WEIGHT_FILE_NAME = "EfficientNet_Weights/EfficientNetBN_tl_best_weights.h5"
IMAGE_HEIGHT = 284
IMAGE_WIDTH = 512

# Créer une 25eme classe qui ne correspond à aucun oiseau
USE_EMPTY_CLASS = False
len_classes = 25 if USE_EMPTY_CLASS else 24
epch = 100
KERNEL_REGULARIZERS = 0.0005
ref_lr = 0.03
ref_batch_size = 1024
dropout = 0.2
batch_size = 5
momentumTest = 0.95
destination_classes = [str(i) for i in range(len_classes)]
### PARAMS spectrogramm_conversion ###
# Lié à IMAGE_WIDTH et IMAGE_HEIGHT
PERCENT_PRINT = 10
# duration_cut -> Découpage des extraits en morceaux de x secondes / 0 = pas de découpage
DURATION_CUT = 10
RANDOM_CUT = True
# Un ratio de 5 permet de sauvegarder 1 enregistrement de la 25eme classe sur 5
# Evite d'avoir une 25eme classe trop chargée en données (sachant que 1 enregistrement contient au minimum 2 extraits)
RATIO_EMPTY_CLASS = 20
PRED_EMPTY_IGNORE_EXTRACT = 0.6
# minimum duration of record
MINIMAL_DURATION = 0.5
MINIMAL_ANIMAL_PRESENCE = 0.5
FREQ_MODIFIER = 0
validation_split = 0.2
USE_DATA_AUGMENTATION = False
RATIO_DATA_AUG = 2


def compute_class_images_count(base_folder: str, class_name: str):
    return sum((1 for _ in os.listdir(f'{base_folder}/{class_name}')))


def compute_all_classes_images_count(base_folder: str):
    return sum((compute_class_images_count(base_folder, c) for c in destination_classes))


def compute_train_images_count():
    return compute_all_classes_images_count(DATASET_TRAIN_DIRECTORY)


def compute_val_images_count():
    return compute_all_classes_images_count(DATASET_VAL_DIRECTORY)


def compute_total_images_count():
    return compute_val_images_count() + compute_train_images_count()


def compute_class_weight():
    class_weight = {}
    for c in destination_classes:
        class_weight[int(c)] = compute_class_images_count(DATASET_TRAIN_DIRECTORY, c)
        class_weight[int(c)] += compute_class_images_count(DATASET_VAL_DIRECTORY, c)

    # Recuperation de la classe comportortant le moins de data
    key_min = min(class_weight.keys(), key=(lambda k: class_weight[k]))
    to_divide = class_weight[key_min]

    for c in destination_classes:
        class_weight[int(c)] /= to_divide

    return class_weight
