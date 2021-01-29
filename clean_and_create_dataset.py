from spectrogamme_conversion import *
from test_spectrogramme_conversion import create_test_spectro_dataset
import shutil

dataset_type = [
    "val_train",
    "test"
]


def clean_dataset(dataset):
    if dataset == dataset_type[0]:
        for c in destination_classes:
            folder = f'{DATASET_TRAIN_DIRECTORY}/{c}'
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    os.remove(f'{folder}/{filename}')

            folder = f'{DATASET_VAL_DIRECTORY}/{c}'
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    os.remove(f'{folder}/{filename}')
    else:
        for folder in os.listdir(DATASET_TEST_DIRECTORY):
            shutil.rmtree(os.path.join(DATASET_TEST_DIRECTORY, folder), ignore_errors=True)


def clean_and_create_dataset(dataset):
    if dataset == dataset_type[0]:
        if not os.path.isdir(DATASET_DIRECTORY):
            os.mkdir(DATASET_DIRECTORY)
        if not os.path.isdir(DATASET_TRAIN_DIRECTORY):
            os.mkdir(DATASET_TRAIN_DIRECTORY)
        if not os.path.isdir(DATASET_VAL_DIRECTORY):
            os.mkdir(DATASET_VAL_DIRECTORY)
    else:
        if not os.path.isdir(DATASET_TEST_DIRECTORY):
            os.mkdir(DATASET_TEST_DIRECTORY)

    print(F"Cleaning {dataset} dataset...")
    clean_dataset(dataset)

    print(F"Creating {dataset} dataset...")

    if dataset == dataset_type[0]:
        create_spectro_dataset()
    else:
        create_test_spectro_dataset()


if __name__ == "__main__":
    # Change dataset_type pour generer test_val ou test
    # 0 == test & val
    # 1 == test
    clean_and_create_dataset(dataset_type[1])
