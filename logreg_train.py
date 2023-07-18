import sys




def train(train_data_path: str):
    print(train_data_path)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(
            "Argument is wrong. Please pass the file path as an argument like `logreg_train.py dataset/dataset_train.csv`."
        )
        exit(1)

    train(sys.argv[1])
