# Python 3.8.5

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image
import torch.nn as nn
import torch
import os

import matplotlib.pyplot as plt
import numpy

# Auto Encoder class
class AutoEncoder(nn.Module):

    def __init__(self,
                 n_features,
                 n_hidden,
                 n_classes,
                 activation):
        super(AutoEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation

        # input layer
        self.layers.append(nn.Linear(n_features, n_hidden))
        # hidden layer
        self.layers.append(nn.Linear(n_hidden, n_hidden))
        # output layer
        self.layers.append(nn.Linear(n_hidden, n_classes))

    def forward(self, inputs):
        h = inputs
        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
        return h

# Dataset class
class UTKFace(Dataset):

    def __init__(self, root, train=False, eval=False, test=False, transform=None):
        self.root = root
        self.data_loc = f"{self.root}{'train/' if train else 'eval/' if eval else 'test/' if test else ''}"
        self.transform = transform
        self.data = self.get_data()
        
    def get_data(self):
        d = {"image_file": []}

        for i in os.listdir(self.data_loc):
            d["image_file"].append(i)

        return d

    def __len__(self):
        return len(self.data['image_file'])

    def __getitem__(self, idx):
        img_loc = self.data["image_file"][idx]
        image = Image.open(os.path.join(self.data_loc, img_loc)).convert("RGB")
        label = Image.open(os.path.join(self.data_loc, img_loc)).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        r, g, b = [], [], []

        for x in image[0]:
            for y in x:
                r.append(y.cpu().item())
        for x in image[1]:
            for y in x:
                g.append(y.cpu().item())
        for x in image[2]:
            for y in x:
                b.append(y.cpu().item())

        img = r + g + b
        return (torch.tensor(img), torch.tensor(img))


        new_image = []
        nr, ng, nb = [], [], []

        width, height = image.shape[1], image.shape[2]


        print(width, height)
        for y in range(height):
            nrx = []
            ngx = []
            nbx = []
            for x in range(width):
                nrx.append(r[y * height + x])
                ngx.append(g[y * height + x])
                nbx.append(b[y * height + x])
            nr.append(nrx)
            ng.append(ngx)
            nb.append(nbx)

        new_image.append(nr)
        new_image.append(ng)
        new_image.append(nb)

        print(torch.tensor(new_image))
        print(image)

        exit()


        return (image, label)


# model parameters
parameters = {'num_features': 3072,
              'num_hidden_nodes': 1000,
              'activation_fnc': F.relu,
              'loss_fnc': nn.MSELoss(),
              'learning_rate': 0.001,
              'num_epochs': 1,
              'gpu': 0}

# CUDA
device = torch.device(f"cuda:{parameters['gpu']}" if torch.cuda.is_available() else "cpu")

# auto encoder model
model = AutoEncoder(parameters['num_features'],             # input feature count
                    parameters['num_hidden_nodes'],         # hidden node count
                    parameters['num_features'],             # output node count
                    parameters['activation_fnc'])           # activation function

# optimizer
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=parameters['learning_rate'])

# data loader
transform = transforms.Compose([transforms.Resize(size=(32,32)), transforms.ToTensor()])
train_data_utkface_target = UTKFace('../Datasets/UTKFace/', train=True, transform=transform)
test_data_utkface_target = UTKFace('../Datasets/UTKFace/', test=True, transform=transform)

train_loader = DataLoader(train_data_utkface_target, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data_utkface_target, batch_size=32, shuffle=False)

# training phase
model.to(device)
model.train()
for epoch in range(parameters['num_epochs']):
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        output = model(inputs)
        loss = parameters['loss_fnc'](output, labels)

        # backward + optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f' [+++] Epoch {epoch + 1} : {loss.cpu().item()}', end='\r')

        old = output[0].cpu().detach().numpy()

        new_image = []
        nr, ng, nb = [], [], []

        width, height = 32, 32
        r, g, b = old[:1024], old[1024:2048], old[2048:]
        for y in range(height):
            nrx = []
            ngx = []
            nbx = []
            for x in range(width):
                nrx.append(r[y * height + x])
                ngx.append(g[y * height + x])
                nbx.append(b[y * height + x])
            nr.append(nrx)
            ng.append(ngx)
            nb.append(nbx)

        new_image.append(nr)
        new_image.append(ng)
        new_image.append(nb)

        aaa = torch.tensor(new_image)
        plt.imshow(aaa.numpy())
        exit()
        print(aaa.shape)    
        arr = numpy.ndarray(aaa)
        #print(arr)
        arr_ = numpy.squeeze(arr)
        plt.imshow(arr_)
        plt.show()
        exit()

torch.save(model.state_dict(), f'./auto-encoder.pth')
