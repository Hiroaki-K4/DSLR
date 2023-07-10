import csv
import sys
import numpy as np
import matplotlib.pyplot as plt


def main(csv_path: str):
    with open(csv_path) as f:
        reader = csv.reader(f, delimiter=',')
        input_data_list = [row for row in reader]

    header = input_data_list[0]
    numerical_feature_start_pos = 6
    numerical_feature_list = header[numerical_feature_start_pos:]
    input_data_list = input_data_list[1:len(input_data_list)]

    for i in range(len(numerical_feature_list)):
        gryffindor_list = []
        hufflepuff_list = []
        ravenclaw_list = []
        slytherin_list = []
        type_0 = "Gryffindor"
        type_1 = "Hufflepuff"
        type_2 = "Ravenclaw"
        type_3 = "Slytherin"
        target_col_num = numerical_feature_start_pos + i
        for data in input_data_list:
            if data[target_col_num] != "":
                if data[1] == type_0:
                    gryffindor_list.append(float(data[target_col_num]))
                elif data[1] == type_1:
                    hufflepuff_list.append(float(data[target_col_num]))
                elif data[1] == type_2:
                    ravenclaw_list.append(float(data[target_col_num]))
                elif data[1] == type_3:
                    slytherin_list.append(float(data[target_col_num]))
                else:
                    print("Wrong name of Hogwarts House.")
                    continue

        plt.hist(gryffindor_list, alpha=0.4, label=type_0, color="red")
        plt.hist(hufflepuff_list, alpha=0.4, label=type_1, color="yellow")
        plt.hist(ravenclaw_list, alpha=0.4, label=type_2, color="blue")
        plt.hist(slytherin_list, alpha=0.4, label=type_3, color="green")
        plt.legend(loc="upper right")
        plt.title(numerical_feature_list[i])
        plt.show()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Argument is wrong. Please pass the file path as an argument like `describe.py dataset/dataset_train.csv`.")
        exit(1)

    main(sys.argv[1])
