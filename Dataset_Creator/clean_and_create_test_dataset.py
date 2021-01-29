from Dataset_Creator.test_spectrogramme_conversion import *


def clean_test_dataset():
    for _, dirs, files in os.walk(DATASET_TEST_DIRECTORY):
        for directory in dirs:
            folder = os.path.join(DATASET_TEST_DIRECTORY, directory)
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    os.remove(os.path.join(folder, filename))


def clean_and_create_dataset():

    if not os.path.isdir(DATASET_DIRECTORY):
        os.mkdir(DATASET_DIRECTORY)
    if not os.path.isdir(DATASET_TEST_DIRECTORY):
        os.mkdir(DATASET_TEST_DIRECTORY)

    print("Cleaning...")
    clean_test_dataset()
    print("Creating spectro dataset...")
    create_test_spectro_dataset()


if __name__ == "__main__":
    clean_and_create_dataset()
