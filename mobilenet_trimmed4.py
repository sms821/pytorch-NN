from math import floor
import torch
import torch.nn as nn
from spiking import spikeRelu

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MobileNet_trimmed4(nn.Module):
    def __init__(self, num_classes=200):
        super(MobileNet_trimmed4, self).__init__()

        #size = 3,56,56 (stride=2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, groups=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        # ofm = 32,28,28

        # dw-conv-1 (stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        # ofm = 32,28,28
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0, groups=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        # ofm = 64,28,28

        # dw-conv-2 (stride=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, groups=64)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        # ofm = 64,14,14
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0, groups=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU(inplace=True)
        # ofm = 128,14,14

        # dw-conv-3 (stride=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128)
        self.bn6 = nn.BatchNorm2d(128)
        self.relu6 = nn.ReLU(inplace=True)
        # ofm = 128,14,14
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.relu7 = nn.ReLU(inplace=True)
        # ofm = 256,14,14

        # dw-conv-4 (stride=2)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, groups=256)
        self.bn8 = nn.BatchNorm2d(256)
        self.relu8 = nn.ReLU(inplace=True)
        # ofm = 256,7,7
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0, groups=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.relu9 = nn.ReLU(inplace=True)
        # ofm = 512,7,7

        # dw-conv-5 (stride=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=512)
        self.bn10 = nn.BatchNorm2d(512)
        self.relu10 = nn.ReLU(inplace=True)
        # ofm = 512,7,7
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, padding=0, groups=1)
        self.bn11 = nn.BatchNorm2d(1024)
        self.relu11 = nn.ReLU(inplace=True)
        # ofm = 1024,7,7

        self.avgpool1 = nn.AvgPool2d(7)
        # ofm = 1024,1,1

        self.fc1 = nn.Linear(1024, num_classes)
        # ofm = 200

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu7(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu8(x)

        x = self.conv9(x)
        x = self.bn9(x)
        x = self.relu9(x)

        x = self.conv10(x)
        x = self.bn10(x)
        x = self.relu10(x)

        x = self.conv11(x)
        x = self.bn11(x)
        x = self.relu11(x)

        x = self.avgpool1(x)

        x = x.view(-1, 1024)
        x = self.fc1(x)

        return x
