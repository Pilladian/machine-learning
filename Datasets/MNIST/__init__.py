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
        path = './MNIST'

        # load partial datasets
        train = MNIST(path, train=True, transform=transform)
        eval  = MNIST(path, eval=True, transform=transform)
        test  = MNIST(path, test=True, transform=transform)

        # create dataloader
        train_loader = DataLoader(train, batch_size=32, shuffle=True)
        eval_loader  = DataLoader(eval, batch_size=32, shuffle=False)
        test_loader  = DataLoader(test, batch_size=32, shuffle=False)


    Explanation:

        File format: <label>_<rand_int>.jpg
    
"""

class MNIST(Dataset):

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