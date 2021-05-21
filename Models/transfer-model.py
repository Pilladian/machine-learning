# Python 3.8.5


# Convolutional Neural Network
def TransferCNN(hidden_nodes, num_classes):
    transferCNN = models.vgg19(pretrained=True)

    # save weights of transferCNN
    for param in transferCNN.parameters():
        param.requires_grad = False

    # classifier of CNN
    model = transferCNN
    model.classifier = nn.Sequential(
                            nn.Linear(25088, 4096),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(4096, hidden_nodes),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.4),
                            nn.Linear(hidden_nodes, num_classes),
                            nn.LogSoftmax(dim=1)
                            )
    return model
