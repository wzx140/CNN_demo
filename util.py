import math
import os

import h5py
import numpy as np
import scipy
from scipy import ndimage


def load_data_set():
    """
    load the data from h5 files
    :return:
    """
    train_data_set = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_data_set["train_set_x"])
    train_set_y_orig = np.array(train_data_set["train_set_y"])

    test_data_set = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_data_set["test_set_x"])
    test_set_y_orig = np.array(test_data_set["test_set_y"])

    classes = np.array(test_data_set["list_classes"])
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def pre_treat(data: np.ndarray, is_x=True, class_num=0):
    """
    normalize the data set or convert y to one-hot
    :param data:
    :param is_x: data is x or y
    :param class_num: number of classes, if you set is_x false, you should give the value of num
    :return:
    """
    if is_x:
        return normalization(data)
    else:
        # convert to one hot
        return np.eye(class_num)[data.reshape(-1)]


def random_mini_batches(x, y, mini_batch_size=64):
    """
    Creates a list of random minibatches from (x, y)

    Arguments:
    :param x: input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    :param y: true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    :param mini_batch_size: size of the mini-batches, integer

    :return: list of synchronous (mini_batch_x, mini_batch_y)
    """

    m = x.shape[0]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_x = x[permutation, :, :, :]
    shuffled_y = y[permutation, :]

    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_x = shuffled_x[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_y = shuffled_y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_x = shuffled_x[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_y = shuffled_y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)

    return mini_batches


def load_pic(path: str = None):
    if path is None or not os.path.isdir(path):
        path = 'image'

    files = os.listdir(path)
    files.sort()
    X = []
    for file in files:
        if file.startswith('data_'):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                image = np.array(ndimage.imread(file_path, flatten=False, mode='RGB'))
                temp = scipy.misc.imresize(image, size=(64, 64))
                temp = normalization(temp)
                X.append(temp)

    # print(X.shape)
    return np.array(X)


def normalization(x):
    """
    normalize the pictures
    :param x:
    :return:
    """
    normalized_x = (x - np.mean(x)) / np.std(x)

    return normalized_x
