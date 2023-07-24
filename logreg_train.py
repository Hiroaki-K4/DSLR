import sys
import csv
from typing import List
import numpy as np
from tqdm import tqdm


def preprocess_input_data(input_data_list: List, target_house_type: str):
    input_data = []
    label_data = []
    for data in input_data_list:
        if data[1] == target_house_type:
            label = 1
        else:
            label = 0
        label_data.append(label)
        numerical_data = data[6:]

        for i in range(len(numerical_data)):
            if numerical_data[i] == "":
                numerical_data[i] = 0.0
            numerical_data[i] = float(numerical_data[i])

        input_data.append(numerical_data)

    if len(input_data) != len(label_data):
        raise RuntimeError("The data size is different between input_data and label_data.")

    return input_data, label_data


def logistic_regression(input_data: List, label_data: List):
    param_arr = np.zeros(len(input_data[0]) + 1)
    learning_rate = 1e-3
    thr = 1e-5
    iterations = 100000
    # for i in tqdm(range(iterations)):



def train(train_data_path: str):
    with open(train_data_path) as f:
        reader = csv.reader(f, delimiter=",")
        input_data_list = [row for row in reader][1:]

    input_data, label_data = preprocess_input_data(input_data_list, "Gryffindor")
    logistic_regression(input_data, label_data)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(
            "Argument is wrong. Please pass the file path as an argument like `python logreg_train.py dataset/dataset_train.csv`."
        )
        exit(1)

    train(sys.argv[1])
