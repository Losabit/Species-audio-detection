import os
import csv
from tensorflow.keras import Model
from matplotlib import image
import numpy as np
from dataset_param import *

length_classes = 24


def average(predicts, *args):
    sum_classes = np.zeros(length_classes, dtype=np.float32)
    for predict in predicts:
        print(predict)
        for i in range(length_classes):
            sum_classes[i] += predict[i]
    return sum_classes / len(predicts)


def random(predicts, *args):
    return predicts[random.randint(0, len(predicts) - 1)]


def higher_than(predicts, limit):
    limit = limit[0]
    higher_prediction = np.zeros(length_classes, dtype=np.float32)
    for predict in predicts:
        for i in range(length_classes):
            if predict[i] >= limit:
                higher_prediction[i] += 1
    total = sum(higher_prediction)
    if total == 0:
        return higher_than(predicts, (limit - limit / 4,))
    return higher_prediction / total


def highest_value(predicts, *args):
    for predict in predicts:
        for i in range(length_classes):
            if predict[i] >= 0.6:
                return


def distribute_empty(pred):
    return_array = np.zeros(len(pred))
    to_distribute = pred[len(pred) - 1]
    for p in range(length_classes):
        return_array[p] = pred[p] + (to_distribute / length_classes)
    return return_array


def predict_and_save_in_submission(model: Model, func, *args):
    with open(os.path.join(DATASET_DIRECTORY, "submission.csv"), mode='w', newline='') as output_csv_file:
        writer = csv.writer(output_csv_file)
        writer.writerow(["recording_id", "s0", "s1", "s2", "s3", "s4", "s5",
                         "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14",
                         "s15", "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23"])
        one_percent = int(sum([len(files) for r, d, files in os.walk(DATASET_TEST_DIRECTORY)]) / 100)
        percent = 0
        line_count = 0
        for _, directories, _ in os.walk(DATASET_TEST_DIRECTORY):
            for directory in directories:
                directory_path = os.path.join(DATASET_TEST_DIRECTORY, directory)
                predictions = np.zeros((0, len_classes), dtype=np.float32)

                for file in os.listdir(directory_path):
                    file_path = os.path.join(directory_path, file)
                    spectro_image = image.imread(file_path)
                    spectro_image = np.expand_dims(spectro_image, axis=0)

                    model_prediction = model.predict(spectro_image)

                    if USE_EMPTY_CLASS and model_prediction[0][len_classes - 1] >= PRED_EMPTY_IGNORE_EXTRACT:
                        model_prediction[0] = np.zeros(len_classes, dtype=np.float32)
                    elif USE_EMPTY_CLASS:
                        model_prediction[0] = distribute_empty(model_prediction[0])

                    predictions = np.concatenate(
                        (predictions,
                         model_prediction),
                        axis=0)

                    if line_count % one_percent == 0:
                        if percent % PERCENT_PRINT == 0:
                            print(str(percent) + "%")
                        percent += 1
                    line_count += 1

                if func is None:
                    writer.writerow(np.insert(predictions[0].astype(np.str), 0, directory, axis=0))
                else:
                    writer.writerow(np.insert(func(predictions, args).astype(np.str), 0, directory, axis=0))
