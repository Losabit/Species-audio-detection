import os, random

DATASET_DIRECTORY = os.path.join(os.getcwd(), 'dataset', 'spectrogram-species-audio-detection')
DATASET_TRAIN_DIRECTORY = os.path.join(DATASET_DIRECTORY, 'train')
DATASET_VAL_DIRECTORY = os.path.join(DATASET_DIRECTORY, 'val')
DATASET_TEST_DIRECTORY = os.path.join(DATASET_DIRECTORY, 'test')
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
# Créer une 25eme classe qui ne correspond à aucun oiseau
USE_EMPTY_CLASS = True
len_classes = 25 if USE_EMPTY_CLASS else 24
epch = 100
KERNEL_REGULARIZERS = 0.0005
ref_lr = 0.03
ref_batch_size = 1024
batch_size = 5
momentumTest = 0.95
destination_classes = [str(i) for i in range(len_classes)]
### PARAMS spectrogramm_conversion ###
# Lié à IMAGE_WIDTH et IMAGE_HEIGHT
PERCENT_PRINT = 10
# duration_cut -> Découpage des extraits en morceaux de x secondes / 0 = pas de découpage
DURATION_CUT = 2
RANDOM_CUT = True
# Un ratio de 5 permet de sauvegarder 1 enregistrement de la 25eme classe sur 5
# Evite d'avoir une 25eme classe trop chargée en données (sachant que 1 enregistrement contient au minimum 2 extraits)
RATIO_EMPTY_CLASS = 48
# minimum duration of record
MINIMAL_DURATION = 0.5
FREQ_MODIFIER = 0
validation_split = 0.3

### PARAMS test_spectrogramm_conversion ###
TEST_DURATION_CUT = 5
TEST_RANDOM_CUT = True
TEST_MINIMAL_DURATION = 1


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
