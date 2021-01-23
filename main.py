import csv
import numpy as np
from matplotlib import image
from dataset_param import *
from utils import load_data


if __name__ == '__main__':
    (data, labels) = load_data()
    print(data.shape)
    print(labels.shape)

