import argparse
import csv

import matplotlib.pyplot as plt


def main(data_file_path, x_item, y_item):
    with open(data_file_path) as f:
        reader = csv.reader(f, delimiter=",")
        input_data_list = [row for row in reader]

    header = input_data_list[0]
    numerical_feature_start_pos = 6
    numerical_feature_list = header[numerical_feature_start_pos:]
    input_data_list = input_data_list[1 : len(input_data_list)]
    if not x_item in numerical_feature_list or not y_item in numerical_feature_list:
        raise RuntimeError("The argument of item is wrong.")

    x_item_idx = header.index(x_item)
    y_item_idx = header.index(y_item)

    type_0 = "Gryffindor"
    type_1 = "Hufflepuff"
    type_2 = "Ravenclaw"
    type_3 = "Slytherin"
    gryffindor_x_list = []
    gryffindor_y_list = []
    hufflepuff_x_list = []
    hufflepuff_y_list = []
    ravenclaw_x_list = []
    ravenclaw_y_list = []
    slytherin_x_list = []
    slytherin_y_list = []
    for data in input_data_list:
        if data[x_item_idx] != "" and data[y_item_idx] != "":
            if data[1] == type_0:
                gryffindor_x_list.append(float(data[x_item_idx]))
                gryffindor_y_list.append(float(data[y_item_idx]))
            elif data[1] == type_1:
                hufflepuff_x_list.append(float(data[x_item_idx]))
                hufflepuff_y_list.append(float(data[y_item_idx]))
            elif data[1] == type_2:
                ravenclaw_x_list.append(float(data[x_item_idx]))
                ravenclaw_y_list.append(float(data[y_item_idx]))
            elif data[1] == type_3:
                slytherin_x_list.append(float(data[x_item_idx]))
                slytherin_y_list.append(float(data[y_item_idx]))
            else:
                print("Wrong name of Hogwarts House.")
                continue

    plt.scatter(gryffindor_x_list, gryffindor_y_list, c="red", label=type_0)
    plt.scatter(hufflepuff_x_list, hufflepuff_y_list, c="yellow", label=type_1)
    plt.scatter(ravenclaw_x_list, ravenclaw_y_list, c="blue", label=type_2)
    plt.scatter(slytherin_x_list, slytherin_y_list, c="green", label=type_3)
    plt.xlabel(x_item)
    plt.ylabel(y_item)
    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file_path")
    parser.add_argument("--x_item")
    parser.add_argument("--y_item")
    args = parser.parse_args()
    main(args.data_file_path, args.x_item, args.y_item)
