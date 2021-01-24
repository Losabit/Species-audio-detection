from dataset_param import *
from utils import *


if __name__ == '__main__':
    (data, labels) = load_data()
    train_data, val_data = split_array(data, 0.8)
    train_labels, val_labels = split_array(labels, 0.8)

