# Python 3.8.5

from torch.utils.data import Dataset
from PIL import Image
import torch
import os

"""
    Usage: 

        import torchvision.transforms as transforms

        transform = transforms.Compose( 
                                [ transforms.Resize(size=(32,32)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
        path = './CIFAR10'

        # load partial datasets
        train = CIFAR10(path, train=True, transform=transform)
        eval  = CIFAR10(path, eval=True, transform=transform)
        test  = CIFAR10(path, test=True, transform=transform)

        # create dataloader
        train_loader = DataLoader(train, batch_size=32, shuffle=True)
        eval_loader  = DataLoader(eval, batch_size=32, shuffle=False)
        test_loader  = DataLoader(test, batch_size=32, shuffle=False)

    Explanation:

        File format: <label>_<rand_int>.png

        | label | meaning
        |---    |---
        | 0     | airplane
        | 1     | automobile
        | 2     | bird
        | 3     | cat
        | 4     | deer
        | 5     | dog
        | 6     | frog
        | 7     | horse
        | 8     | ship
        | 9     | truck
    
"""

class CIFAR10(Dataset):

    def __init__(self, root, train=False, eval=False, test=False, transform=None):
        self.root = root
        self.data_loc = f"{self.root}{'train/' if train else 'eval/' if eval else 'test/' if test else ''}"
        self.transform = transform
        self.data = self.get_data()
        
    def get_data(self):
        d = {"image_file": [], "label": []}

        for i in os.listdir(self.data_loc):
            l = i.split('_')
            d["image_file"].append(i)
            d["label"].append(l[0]) # airplane, ship, ...

        return d

    def __len__(self):
        return len(self.data['image_file'])

    def __getitem__(self, idx):
        img_loc = self.data["image_file"][idx]
        image = Image.open(os.path.join(self.data_loc, img_loc)).convert("RGB")
        label = torch.tensor(float(self.data["label"][idx]))

        if self.transform is not None:
            image = self.transform(image)

        return (image, label)