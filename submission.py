import os
import csv
from tensorflow.keras import Model
from matplotlib import image
import numpy as np
from dataset_param import *


def average(predicts):
    sum_classes = np.zeros(len_classes, dtype=np.float32)
    for predict in predicts:
        for i in range(len_classes):
            sum_classes[i] += predict[i]
    return sum_classes / len(predicts)


def predict_and_save_in_submission(model: Model, func):
    with open(os.path.join(DATASET_DIRECTORY, "submission.csv"), mode='w', newline='') as output_csv_file:
        writer = csv.writer(output_csv_file)
        writer.writerow(["recording_id", "s0", "s1", "s2", "s3", "s4", "s5",
                         "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14",
                         "s15", "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23"])
        for _, directories, _ in os.walk(DATASET_TEST_DIRECTORY):
            for directory in directories:
                directory_path = os.path.join(DATASET_TEST_DIRECTORY, directory)
                predictions = np.zeros((0, len_classes), dtype=np.float32)
                for file in os.listdir(directory_path):
                    file_path = os.path.join(directory_path, file)
                    spectro_image = image.imread(file_path)
                    spectro_image = np.expand_dims(spectro_image, axis=0)
                    predictions = np.concatenate((predictions, model.predict(spectro_image)), axis=0)

                if func is None:
                    writer.writerow(np.insert(predictions[0].astype(np.str), 0, directory, axis=0))
                else:
                    writer.writerow(np.insert(func(predictions).astype(np.str), 0, directory, axis=0))
