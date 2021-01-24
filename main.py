from dataset_param import *
from utils import *


if __name__ == '__main__':
    (data, labels) = load_data()
    print(data.shape)
    print(labels.shape)
    print(split_array(data, 500).shape)

