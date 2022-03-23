import numpy as np
import torch as T
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch.nn as nn

import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils import load_csv
from anomaly_net import Anomaly_Net
from batch import Batch

dataset = "dataset"
archive = "archive"
test = "test"
train = "train"
good = "good"
not_good = "not-good"
np_txt = "np_text"

csv_path = "csv"
csv_dir_good = "good.csv"
csv_dir_not_good = "not_good.csv"
csv_dir_test = "test.csv"

cwd_parent = os.path.dirname(os.getcwd())
test_dir = os.path.join(os.path.abspath(cwd_parent), dataset, archive, test)
train_good_dir = os.path.join(os.path.abspath(cwd_parent), dataset, archive, train, good)
train_not_good_dir = os.path.join(os.path.abspath(cwd_parent), dataset, archive, train, not_good)

csv_train_good_dir = os.path.join(os.path.abspath(cwd_parent), dataset, archive, csv_path, csv_dir_good)
csv_train_not_good_dir = os.path.join(os.path.abspath(cwd_parent), dataset, archive, csv_path, csv_dir_not_good)
csv_test_dir = os.path.join(os.path.abspath(cwd_parent), dataset, archive, csv_path, csv_dir_test)

model_path = "entire_model_lr_tanh_2.pt"
# model_path = "entire_model_RELU.pt"

def main():
    print("Autoencoder for MVtec screws")

    T.manual_seed(1)
    np.random.seed(1)

    # load data
    train_good_data = load_csv(csv_train_good_dir, False)
    train_not_good_data = load_csv(csv_train_not_good_dir, False)

    # print("this is train_good_data", train_good_data)
    print("this is train_not_good_data", train_not_good_data)

    # combine data
    # train_data = np.append(train_good_data, train_not_good_data, axis=0)

    # normal data (no anomaly)
    train_data = train_good_data

    print("this is train_data", train_data)
    print("this is length of train_data", len(train_data))

    # split labels from training data
    data_x = train_data[:, 1:]
    labels = train_data[:, :1]

    # resize image

    # normalize_data by dividing by pixel max value
    norm_x = data_x / 255

    # create network
    model = Anomaly_Net()

    # train model
    model.train()
    batch_size = 20
    loss_func = nn.MSELoss()

    # lower learning rate more?
    optimizer = T.optim.Adam(model.parameters(), lr = 0.001)

    batch_item = Batch(num_items=len(norm_x), batch_size=batch_size, seed=1)

    max_epochs = 1000

    print("start training")

    for epoch in range(0, max_epochs):
        if epoch > 0 and epoch % (max_epochs/10) == 0:
            print("epoch = %6d" % epoch, end="")
            print("  prev batch loss = %7.4f" % loss_obj.item())

        for current_batch in batch_item:
            X = T.Tensor(norm_x[current_batch])
            optimizer.zero_grad()
            output = model(X)
            loss_obj = loss_func(output, X)
            loss_obj.backward()
            optimizer.step()

    print("training finished")
    
    print("saving model")
    T.save(model, model_path)

def predict_test():
    print("this is the predict method")

    train_good_data = load_csv(csv_train_good_dir, False)
    test_data = load_csv(csv_test_dir, False)
    train_data = test_data

    # split labels from training data
    data_x = train_data[:, 1:]
    labels = train_data[:, :1]

    # test_data is unlabeled, so don't split labels
    data_x = test_data

    norm_x = data_x / 255
    H, W = 256, 256

    model = T.load(model_path)
    model = model.eval()

    X = T.Tensor(norm_x)
    Y = model(X)
    N = len(data_x)

    loss_collect = []

    loss_base = nn.MSELoss(reduction="mean")

    for i in range(N):
        loss_error = loss_base(X[i], Y[i])
        print("this is loss", loss_error)
        print("this is loss.item", loss_error.item())
        loss_collect.append(loss_error.item())

    loss_plot = []

    for i in loss_collect:
        loss_plot.append((i, i))
    
    plt.figure()

    plt.scatter(*zip(*loss_plot))
    plt.axvline(0.03, 0.0, 0.1)

    plt.legend()
    # plt.figure()

    # lower_threshold = 0.0
    # upper_threshold = 0.03
    # plt.figure(figsize=(12,6))
    # plt.title('Loss Distribution')
    # sns.distplot(loss_collect, bins=100, kde=True, color='blue')
    # plt.axvline(upper_threshold, 0.0, 10, color='r')
    # plt.axvline(lower_threshold, 0.0, 10, color='b')

    plt.show()


def predict_train():
    print("this is anomaly detection predicting all training data")
    train_good_data = load_csv(csv_train_good_dir, False)
    train_not_good_data = load_csv(csv_train_not_good_dir, False)

    # combine all train data
    train_data = np.append(train_good_data, train_not_good_data, axis=0)

    data_x = train_data[:, 1:]
    labels = train_data[:, :1]

    norm_x = data_x / 255
    H, W = 256, 256

    model = T.load(model_path)
    model = model.eval()

    X = T.Tensor(norm_x)
    Y = model(X)
    N = len(data_x)

    loss_collect = []

    loss_base = nn.MSELoss(reduction="mean")

    print("this is length of N", N)

    for i in range(N):
        loss_error = loss_base(X[i], Y[i])
        # print("this is loss", loss_error)
        # print("this is loss.item", loss_error.item())
        loss_collect.append(loss_error.item())

    upper_threshold = 0.03

    # prediction says anomaly, and it is anomaly
    true_positive = 0
    # prediction says anoamaly, but not anomaly
    false_positive = 0
    # prediction says not anomaly, and is not anomaly
    true_negative = 0
    # prediction says not anomaly, but is anomaly
    false_negative = 0
    
    total_anomaly = 0

    collect_indices = []

    for i in range(N):
        if loss_collect[i] >= upper_threshold:
            total_anomaly += 1
            
            if labels[i] == 0:
                true_positive += 1
            else:
                false_positive += 1
        
        else:
            if labels[i] == 0:
                false_negative += 1
            else:
                true_negative += 1
    
    print('[TP] {}\t[FP] {}\t[MISSED] {}'.format(true_positive, false_positive, total_anomaly - true_positive))
    print('[TN] {}\t[FN] {}'.format(true_negative, false_negative))


def sample_reconstruction():
    print("this is the sample reconstruction method")

    train_good_data = load_csv(csv_train_good_dir, False)
    train_not_good_data = load_csv(csv_train_not_good_dir, False)
    # test_data = load_csv(csv_test_dir)

    # combine data
    # train_data = np.append(train_good_data, train_not_good_data, axis=0)

    # normal data (no anomaly)
    train_data = train_good_data

    # split labels from training data
    data_x = train_data[:, 1:]
    labels = train_data[:, :1]

    norm_x = data_x / 255
    H, W = 256, 256

    print("this is before constructing model data_x", data_x)
    # model = Anomaly_Net()
    # model.load_state_dict(T.load(model_path))

    model = T.load(model_path)
    model = model.eval()

    X = T.Tensor(norm_x)
    Y = model(X)
    N = len(data_x)

    # plot test image
    arr = data_x[0]
    arr = arr.reshape((256, 256)).astype('float32')

    cv2.imwrite("train_good_first_image.png", arr)

    # plot test reconstruction
    # reconstruction_arr = Y[data_x[0]]

    reconstruction_arr = Y[0]
    print("this is length of reconstruction arr", len(reconstruction_arr))

    print("this is reconstruction_arr", reconstruction_arr)

    reconstruction_arr = reconstruction_arr.detach().cpu().numpy()
    reconstruction_arr = reconstruction_arr * 255

    print("this is reconstruction arr pixel value", reconstruction_arr)
    print("values less than 1", reconstruction_arr[reconstruction_arr < 1])
    print("their indices are ", np.nonzero(reconstruction_arr < 1))

    reconstruction_arr = np.around(reconstruction_arr).astype(int)

    # indices_zero = reconstruction_arr == 0
    # reconstruction_arr[indices_zero] = 1

    reconstruction_arr.astype('float32')


    reconstruction_arr = reconstruction_arr.reshape((256, 256))

    # round before casting to int?
    # reconstruction_arr = np.around(reconstruction_arr).astype(int).astype('float32')

    cv2.imwrite("train_good_first_image_reconstruction.png", reconstruction_arr)

    print("this is data_x[0]", data_x[0])
    print("this is reconstruction_arr", reconstruction_arr)


def check_GPU():
    print("is gpu available: ", T.cuda.is_available())

if __name__ == "__main__":
    print("this is anomaly_detection_main")
    # main()
    # sample_reconstruction()
    # check_GPU()
    # predict_test()
    predict_train()