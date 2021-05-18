# Python 3.8.5


import pandas
import os


def create_csv_example(root):
    data = pandas.DataFrame(columns=["img_file_name", "gender"])
    data["img_file_name"] = os.listdir(root)

    for ind, filename in enumerate(os.listdir(root)):
        if "female" in filename:
            data["gender"][ind] = 0
        elif "male" in filename:
            data["gender"][ind] = 1
        else:
            data["gender"][ind] = 2

    data.to_csv("data.csv", index=False, header=True)
