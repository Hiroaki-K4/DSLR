import csv
import sys

import matplotlib.pyplot as plt


def main(csv_path: str):
    with open(csv_path) as f:
        reader = csv.reader(f, delimiter=",")
        input_data_list = [row for row in reader]

    header = input_data_list[0]
    numerical_feature_start_pos = 6
    numerical_feature_list = header[numerical_feature_start_pos:]
    input_data_list = input_data_list[1 : len(input_data_list)]

    col_num = 5
    if len(numerical_feature_list) % col_num == 0:
        graph_row = int(len(numerical_feature_list) / col_num)
    else:
        graph_row = int(len(numerical_feature_list) / col_num) + 1
    fig = plt.figure(figsize=(16, 9))

    type_0 = "Gryffindor"
    type_1 = "Hufflepuff"
    type_2 = "Ravenclaw"
    type_3 = "Slytherin"
    for i in range(len(numerical_feature_list)):
        gryffindor_list = []
        hufflepuff_list = []
        ravenclaw_list = []
        slytherin_list = []
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

        graph = fig.add_subplot(graph_row, col_num, i + 1)
        graph.hist(gryffindor_list, alpha=0.4, label=type_0, color="red")
        graph.hist(hufflepuff_list, alpha=0.4, label=type_1, color="yellow")
        graph.hist(ravenclaw_list, alpha=0.4, label=type_2, color="blue")
        graph.hist(slytherin_list, alpha=0.4, label=type_3, color="green")
        graph.legend(loc="upper right", fontsize="7")
        graph.set_title(numerical_feature_list[i], fontsize=10)

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Argument is wrong. Please pass the file path as an argument like `describe.py dataset/dataset_train.csv`."
        )
        exit(1)

    main(sys.argv[1])
