import argparse
import csv
import math
from typing import List

import numpy as np
import yaml


def normalize_range(data_arr):
    min_list = []
    max_list = []
    remv_num = 1
    for i in range(remv_num, data_arr.shape[1]):
        col = data_arr[:, i]
        min_list.append(min(col))
        max_list.append(max(col))

    for j in range(data_arr.shape[0]):
        for k in range(remv_num, data_arr.shape[1]):
            data_arr[j, k] = (data_arr[j, k] - min_list[k - 1]) / (
                max_list[k - 1] - min_list[k - 1]
            )

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
        raise RuntimeError(
            "The data size is different between input_data and label_data."
        )

    return input_data, label_data


def calculate_sigmoid(params, inputs):
    return 1 / (1 + math.e ** ((-1) * np.dot(params.T, inputs)))


def calculate_cost_func(input_data: List, label_data: List, params):
    cost_sum = 0
    for i in range(len(input_data)):
        cost_sum += (-1) * (
            label_data[i] * math.log(calculate_sigmoid(params, np.array(input_data[i])))
            + (1 - label_data[i])
            * math.log(1 - calculate_sigmoid(params, np.array(input_data[i])))
        )

    return cost_sum / len(input_data)


def calculate_cost_derivative(input_data, label_data, param_arr, idx):
    der_sum = 0
    for i in range(len(input_data)):
        der_sum += (
            calculate_sigmoid(param_arr, np.array(input_data[i])) - label_data[i]
        ) * input_data[i][idx]

    return der_sum / len(input_data)


def update_params(param_arr, lr_rate, input_data, label_data):
    new_params = []
    for i in range(param_arr.shape[0]):
        cost_der = calculate_cost_derivative(input_data, label_data, param_arr, i)
        new_param = param_arr[i] - lr_rate * cost_der
        new_params.append(new_param)

    return np.array(new_params)


def logistic_regression(input_data: List, label_data: List):
    param_arr = np.zeros(len(input_data[0]))
    lr_rate = 1e-3
    thr = 1e-5
    iterations = 10000
    prev_cost = 100000
    for i in range(iterations):
        param_arr = update_params(param_arr, lr_rate, input_data, label_data)
        cost = calculate_cost_func(input_data, label_data, param_arr)
        if cost < thr or abs(prev_cost - cost) < thr:
            break
        if i % 100 == 0 and i > 0:
            print("Iteration: {0} Cost: {1}".format(i, cost))

        prev_cost = cost

    print("Iteration: {0} Final Cost: {1}".format(i, cost))

    return param_arr


def save_parameters(param_list, house_list, output_param_path):
    if len(param_list) != len(house_list):
        raise RuntimeError("The length of parameter list and house list is different")

    data = {}
    for i in range(len(house_list)):
        data[house_list[i]] = param_list[i]

    with open(output_param_path, "w") as output_file:
        yaml.dump(data, output_file)


def train(train_data_path: str, output_param_path: str):
    with open(train_data_path) as f:
        reader = csv.reader(f, delimiter=",")
        input_data_list = [row for row in reader][1:]

    house_list = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    param_list = []
    for i in range(len(house_list)):
        input_data, label_data = preprocess_input_data(input_data_list, house_list[i])
        input_data = np.array(input_data)
        norm_input_data = normalize_range(input_data)
        param_arr = logistic_regression(norm_input_data, label_data)
        param_list.append(param_arr.tolist())

    save_parameters(param_list, house_list, output_param_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path")
    parser.add_argument("--output_param_path")
    args = parser.parse_args()

    train(args.train_data_path, args.output_param_path)
