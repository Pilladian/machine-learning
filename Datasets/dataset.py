# Python 3.8.5


from torch.utils.data import Dataset
import pandas
import os
from PIL import Image
import torch


# create csv file for dataset
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


# Example: Images
class ImageDataset(Dataset):

    def __init__(self, root, train=False, eval=False, test=False, transform=None):
        self.root = root
        self.data_loc = f"{root.split('/')[0]}/{'train/' if train else 'eval/' if eval else 'test/' if test else ''}"
        self.data = pandas.read_csv(f'{self.data_loc}/data.csv')
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_loc = self.data.iloc[idx, 0]
        image = Image.open(os.path.join(self.data_loc, img_loc)).convert("RGB")
        label = torch.tensor(float(self.data.iloc[idx, 1]))

        if self.transform is not None:
            image = self.transform(image)

        return (image, label)
