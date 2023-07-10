import csv
import sys


def main(csv_path: str):
    with open(csv_path) as f:
        reader = csv.reader(f, delimiter=',')
        input_data_list = [row for row in reader]

    header = input_data_list[0]
    numerical_feature_list = header[6:]
    # print(header)
    print(numerical_feature_list)
    input_data_list = input_data_list[1:len(input_data_list)]
    # print(input_data_list[0])

    for i in range(len(numerical_feature_list)):
        col_num = i + 6
        print("col_num: ", col_num)
        count = 0
        min_value = sys.float_info.max
        max_value = sys.float_info.min
        print("min_value: ", min_value)
        print("max_value: ", max_value)
        for data in input_data_list:
            print(data[col_num])
            input()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Argument is wrong. Please pass the file path as an argument like `describe.py dataset/dataset_train.csv`.")
        exit(1)
    main(sys.argv[1])
