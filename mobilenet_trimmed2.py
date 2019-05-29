from math import floor
import torch
import torch.nn as nn
from spiking import spikeRelu

class MobileNet_trimmed2(nn.Module):
    def __init__(self):
        super(MobileNet_trimmed2, self).__init__()

        #size = 3,56,56
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(0.1)
        # ofm = 32,56,56

        # dw-conv-1 (stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(0.1)
        # ofm = 32,56,56
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        self.drop3 = nn.Dropout(0.1)
        # ofm = 64,56,56

        # dw-conv-2 (stride=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, groups=64, bias=False)
        self.relu4 = nn.ReLU(inplace=True)
        self.drop4 = nn.Dropout(0.1)
        # ofm = 64,28,28
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.relu5 = nn.ReLU(inplace=True)
        self.drop5 = nn.Dropout(0.1)
        # ofm = 128,28,28

        # dw-conv-3 (stride=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128, bias=False)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout(0.1)
        # ofm = 128,28,28
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout(0.1)
        # ofm = 256,28,28

        # dw-conv-4 (stride=2)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, groups=256, bias=False)
        self.relu8 = nn.ReLU(inplace=True)
        self.drop8 = nn.Dropout(0.1)
        # ofm = 256,14,14
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.relu9 = nn.ReLU(inplace=True)
        self.drop9 = nn.Dropout(0.1)
        # ofm = 512,14,14

        # dw-conv-5 (stride=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=512, bias=False)
        self.relu10 = nn.ReLU(inplace=True)
        self.drop10 = nn.Dropout(0.1)
        # ofm = 512,14,14
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.relu11 = nn.ReLU(inplace=True)
        self.drop11 = nn.Dropout(0.1)
        # ofm = 1024,14,14

        # dw-conv-6 (stride=2)
        self.conv12 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1, groups=1024, bias=False)
        self.relu12 = nn.ReLU(inplace=True)
        self.drop12 = nn.Dropout(0.1)
        # ofm = 1024,7,7
        self.conv13 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.relu13 = nn.ReLU(inplace=True)
        # ofm = 2048,7,7

        self.avgpool1 = nn.AvgPool2d(7)
        # ofm = 2048,1,1

        self.fc1 = nn.Linear(2048, 200, bias=False)
        # ofm = 200

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.drop3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.drop4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.drop5(x)

        x = self.conv6(x)
        x = self.relu6(x)
        x = self.drop6(x)

        x = self.conv7(x)
        x = self.relu7(x)
        x = self.drop7(x)

        x = self.conv8(x)
        x = self.relu8(x)
        x = self.drop8(x)

        x = self.conv9(x)
        x = self.relu9(x)
        x = self.drop9(x)

        x = self.conv10(x)
        x = self.relu10(x)
        x = self.drop10(x)

        x = self.conv11(x)
        x = self.relu11(x)
        x = self.drop11(x)

        x = self.conv12(x)
        x = self.relu12(x)
        x = self.drop12(x)

        x = self.conv13(x)
        x = self.relu13(x)

        x = self.avgpool1(x)

        x = x.view(-1, 2048)
        x = self.fc1(x)

        return x
