import sys


def main():
    print("ok")


if __name__ == '__main__':
    # print(len(sys.argv))
    if len(sys.argv) != 2:
        print("Argument is wrong. Please pass the file path as an argument like `describe.py dataset/dataset_train.csv`.")
        exit(1)
    main()
