import sys
import csv
from typing import List
import numpy as np
import math
from tqdm import tqdm


def normalize_range(data_arr):
    min_list = []
    max_list = []
    remv_num = 1
    for i in range(remv_num, data_arr.shape[1]):
        col = data_arr[ :,i]
        min_list.append(min(col))
        max_list.append(max(col))

    for j in range(data_arr.shape[0]):
        for k in range(remv_num, data_arr.shape[1]):
            # norm_data = (data_list[i] - min_data) / (max_data - min_data)
            data_arr[j, k] = (data_arr[j, k] - min_list[k - 1]) / (max_list[k - 1] - min_list[k - 1])

    return data_arr


def get_input_means(input_data_list, not_num_feat):
    input_mean_list = np.zeros(len(input_data_list[0]) - not_num_feat)
    count_list = np.zeros(len(input_data_list[0]) - not_num_feat)
    for data in input_data_list:
        numerical_data = data[not_num_feat:]
        for i in range(len(numerical_data)):
            if numerical_data[i] == "":
                continue
            input_mean_list[i] += np.float64(numerical_data[i])
            count_list[i] += 1

    input_mean_list = input_mean_list / count_list

    return input_mean_list


def preprocess_input_data(input_data_list: List, target_house_type: str):
    not_num_feat = 6
    input_mean_list = get_input_means(input_data_list, not_num_feat)
    input_data = []
    label_data = []
    for data in input_data_list:
        if data[1] == target_house_type:
            label = 1
        else:
            label = 0
        label_data.append(label)
        numerical_data = data[not_num_feat:]
        numerical_data.insert(0, 1)

        for i in range(len(numerical_data)):
            if numerical_data[i] == "":
                numerical_data[i] = input_mean_list[i + 1]
            numerical_data[i] = float(numerical_data[i])

        input_data.append(numerical_data)

    if len(input_data) != len(label_data):
        raise RuntimeError("The data size is different between input_data and label_data.")

    return input_data, label_data


def calculate_sigmoid(params, inputs):
    # print("(-1) * np.dot(params.T, inputs): ", (-1) * np.dot(params.T, inputs))
    # input()
    # print("sigmoid: ", np.dot(params.T, inputs).shape)
    # print("sig1: ", (-1) * np.dot(params.T, inputs))
    # print("sig2: ", 1 + math.e ** ((-1) * np.dot(params.T, inputs)))
    # print(1 / (1 + math.e ** ((-1) * np.dot(params.T, inputs))))
    return 1 / (1 + math.e ** ((-1) * np.dot(params.T, inputs)))
    # return 1 / (1 + math.e ** ((-1) * np.dot(params, inputs.T)))


def calculate_cost_func(input_data: List, label_data: List, params):
    cost_sum = 0
    for i in range(len(input_data)):
        pred = math.log(calculate_sigmoid(params, np.array(input_data[i])))
        cost_sum += label_data[i] * pred + (1 - label_data[i]) * pred

    return (-1) * cost_sum / len(input_data)


def calculate_cost_derivative(input_data, label_data, param_arr, idx):
    der_sum = 0
    for i in range(len(input_data)):
        print("sig: ", calculate_sigmoid(param_arr, np.array(input_data[i])))
        print("label: ", label_data[i])
        # input()
        der_sum += (calculate_sigmoid(param_arr, np.array(input_data[i])) - label_data[i]) * input_data[i][idx]

    return der_sum / len(input_data)


def update_params(param_arr, lr_rate, input_data, label_data):
    new_params = []
    for i in range(param_arr.shape[0]):
        cost_der = calculate_cost_derivative(input_data, label_data, param_arr, i)
        # TODO: Understand why cost increase
        new_param = param_arr[i] - lr_rate * cost_der
        # print(new_param)
        # input()
        new_params.append(new_param)

    return np.array(new_params)


def logistic_regression(input_data: List, label_data: List):
    param_arr = np.zeros(len(input_data[0]))
    # param_arr = np.ones(len(input_data[0]))
    lr_rate = 1e-1
    thr = 1e-5
    iterations = 100000
    # p = np.array([1.2, 1.3])
    # inp = np.array([0.1, 0.001])
    # ans = calculate_sigmoid(p, inp)
    # print(ans)
    for i in tqdm(range(iterations)):
        param_arr = update_params(param_arr, lr_rate, input_data, label_data)
        cost = calculate_cost_func(input_data, label_data, param_arr)
        print("cost: ", cost)
        if cost < thr:
            print("Cost is lower than threshold")
            break
        # input()


def train(train_data_path: str):
    with open(train_data_path) as f:
        reader = csv.reader(f, delimiter=",")
        input_data_list = [row for row in reader][1:]

    input_data, label_data = preprocess_input_data(input_data_list, "Gryffindor")
    input_data = np.array(input_data)
    norm_input_data = normalize_range(input_data)
    logistic_regression(norm_input_data, label_data)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(
            "Argument is wrong. Please pass the file path as an argument like `python logreg_train.py dataset/dataset_train.csv`."
        )
        exit(1)

    train(sys.argv[1])
