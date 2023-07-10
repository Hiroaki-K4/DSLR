import csv
import sys
import math


def main(csv_path: str):
    with open(csv_path) as f:
        reader = csv.reader(f, delimiter=',')
        input_data_list = [row for row in reader]

    header = input_data_list[0]
    numerical_feature_start_pos = 6
    numerical_feature_list = header[numerical_feature_start_pos:]
    input_data_list = input_data_list[1:len(input_data_list)]


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Argument is wrong. Please pass the file path as an argument like `describe.py dataset/dataset_train.csv`.")
        exit(1)

    main(sys.argv[1])
