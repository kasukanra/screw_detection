import numpy as np
import torch as T
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch.nn as nn

import cv2

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

model_path = "entire_model_lr_tanh.pt"

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
    batch_size = 5
    loss_func = nn.MSELoss()

    # lower learning rate more?
    optimizer = T.optim.Adam(model.parameters(), lr = 0.001)

    batch_item = Batch(num_items=len(norm_x), batch_size=batch_size, seed=1)

    max_epochs = 2000

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

def sample_reconstruction():
    print("this is predict method")

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


    reconstruction_arr = reconstruction_arr.reshape((256, 256))
    reconstruction_arr = reconstruction_arr.astype(int).astype('float32')

    cv2.imwrite("train_good_first_image_reconstruction.png", reconstruction_arr)


def check_GPU():
    print("is gpu available: ", T.cuda.is_available())

if __name__ == "__main__":
    print("this is anomaly_detection_main")
    main()
    # sample_reconstruction()
    # check_GPU()