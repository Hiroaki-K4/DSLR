import argparse
import csv

import numpy as np
import yaml

from logreg_train import calculate_sigmoid, normalize_range


def predict(
    norm_data_arr, gry_param, huf_param, rav_param, sly_param, min_arr, max_arr
):
    result = []
    for i in range(norm_data_arr.shape[0]):
        input_arr = norm_data_arr[i, :]
        gry_bin = calculate_sigmoid(gry_param, input_arr)
        huf_bin = calculate_sigmoid(huf_param, input_arr)
        rav_bin = calculate_sigmoid(rav_param, input_arr)
        sly_bin = calculate_sigmoid(sly_param, input_arr)
        house_bins = [gry_bin, huf_bin, rav_bin, sly_bin]
        house_idx = house_bins.index(max(house_bins))
        if house_idx == 0:
            result.append([i, "Gryffindor"])
        elif house_idx == 1:
            result.append([i, "Hufflepuff"])
        elif house_idx == 2:
            result.append([i, "Ravenclaw"])
        elif house_idx == 3:
            result.append([i, "Slytherin"])

    return result


def preprocess_for_prediction(data_list, mean_arr, not_num_feat, remv_num):
    input_data = []
    for data in data_list:
        numerical_data = data[not_num_feat:]
        numerical_data.insert(0, 1)

        for i in range(remv_num, len(numerical_data)):
            if numerical_data[i] == "":
                numerical_data[i] = mean_arr[i - 1]
            numerical_data[i] = float(numerical_data[i])

        input_data.append(numerical_data)

    return np.array(input_data)


def evaluate(pred_result, truth_data_list):
    if len(pred_result) != len(truth_data_list):
        raise RuntimeError("The lengths of prediction result and truth data are wrong")
    correct = 0
    error = 0
    for i in range(len(truth_data_list)):
        if pred_result[i][1] == truth_data_list[i][1]:
            correct += 1
        else:
            error += 1

    return correct / (correct + error)


def main(test_data_path: str, truth_data_path: str, param_path: str):
    with open(test_data_path) as f:
        reader = csv.reader(f, delimiter=",")
        test_data_list = [row for row in reader][1:]

    with open(truth_data_path) as f:
        reader = csv.reader(f, delimiter=",")
        truth_data_list = [row for row in reader][1:]

    if len(test_data_list) != len(truth_data_list):
        raise RuntimeError("The lengths of test data and truth data are wrong")

    with open(param_path, "r") as stream:
        param = yaml.safe_load(stream)

    gry_param = np.array(np.float64(param["Gryffindor"]))
    huf_param = np.array(np.float64(param["Hufflepuff"]))
    rav_param = np.array(np.float64(param["Ravenclaw"]))
    sly_param = np.array(np.float64(param["Slytherin"]))

    min_arr = np.array(np.float64(param["min_list"]))
    max_arr = np.array(np.float64(param["max_list"]))
    mean_arr = np.array(np.float64(param["mean_list"]))

    not_num_feat = 6
    remv_num = 1
    test_data = preprocess_for_prediction(
        test_data_list, mean_arr, not_num_feat, remv_num
    )
    norm_input_data = normalize_range(test_data, min_arr, max_arr, remv_num)
    pred_result = predict(
        norm_input_data, gry_param, huf_param, rav_param, sly_param, min_arr, max_arr
    )

    accuracy = evaluate(pred_result, truth_data_list)
    print("Accuracy: {0}%".format(accuracy * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path")
    parser.add_argument("--truth_data_path")
    parser.add_argument("--param_path")
    args = parser.parse_args()

    main(args.test_data_path, args.truth_data_path, args.param_path)
