import torch
import torch.nn as nn


class MyNet_in(nn.Module):
    def __init__(self):
        super(MyNet_in, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2)
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(192, 384, kernel_size=3, padding=1),
            # nn.ReLU()
            # nn.Conv2d(384, 256, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256 * 4 * 4, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, NUM_CLASSES)
        # )

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), 256 * 4 * 4)
        # x = self.classifier(x)

        return x

class MyNet_hidden(nn.Module):
    def __init__(self):
        super(MyNet_hidden, self).__init__()
        self.features = nn.Sequential(
            # nn.Conv2d(3, 64, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1)
            # nn.ReLU(),
            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256 * 4 * 4, 4096),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU()
        #     nn.Linear(4096, num_classes)
        # )

    def forward(self, x):
        x = self.features(x)        
        # x = x.view(x.size(0), 256 * 4 * 4)
        # x = self.classifier(x)

        return x

class MyNet_out(nn.Module):
    def __init__(self, NUM_CLASSES):
        super(MyNet_out, self).__init__()
        self.features = nn.Sequential(
            # nn.Conv2d(3, 64, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(64, 192, kernel_size=5, padding=2),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(192, 384, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 4 * 4)
        x = self.classifier(x)

        return x