import numpy as np
import torch as T
import matplotlib.pyplot as plt
import os

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

cwd_parent = os.path.dirname(os.getcwd())
test_dir = os.path.join(os.path.abspath(cwd_parent), dataset, archive, test)
train_good_dir = os.path.join(os.path.abspath(cwd_parent), dataset, archive, train, good)
train_not_good_dir = os.path.join(os.path.abspath(cwd_parent), dataset, archive, train, not_good)

csv_train_good_dir = os.path.join(os.path.abspath(cwd_parent), dataset, archive, csv_path, csv_dir_good)
csv_train_not_good_dir = os.path.join(os.path.abspath(cwd_parent), dataset, archive, csv_path, csv_dir_not_good)





if __name__ == "__main__":
    print("this is anomaly_detection_main")