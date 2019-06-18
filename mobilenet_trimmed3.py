from math import floor
import torch
import torch.nn as nn
from spiking import spikeRelu

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MobileNet_trimmed3(nn.Module):
    def __init__(self):
        super(MobileNet_trimmed3, self).__init__()

        #size = 3,56,56 (stride=2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        # ofm = 32,28,28

        # dw-conv-1 (stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        # ofm = 32,28,28
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        # ofm = 64,28,28

        # dw-conv-2 (stride=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, groups=64, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        # ofm = 64,14,14
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU(inplace=True)
        # ofm = 128,14,14

        # dw-conv-3 (stride=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128, bias=False)
        self.bn6 = nn.BatchNorm2d(128)
        self.relu6 = nn.ReLU(inplace=True)
        # ofm = 128,14,14
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.bn7 = nn.BatchNorm2d(256)
        self.relu7 = nn.ReLU(inplace=True)
        # ofm = 256,14,14

        # dw-conv-4 (stride=2)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, groups=256, bias=False)
        self.bn8 = nn.BatchNorm2d(256)
        self.relu8 = nn.ReLU(inplace=True)
        # ofm = 256,7,7
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.bn9 = nn.BatchNorm2d(512)
        self.relu9 = nn.ReLU(inplace=True)
        # ofm = 512,7,7

        # dw-conv-5 (stride=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=512, bias=False)
        self.bn10 = nn.BatchNorm2d(512)
        self.relu10 = nn.ReLU(inplace=True)
        # ofm = 512,7,7
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.bn11 = nn.BatchNorm2d(1024)
        self.relu11 = nn.ReLU(inplace=True)
        # ofm = 1024,7,7

        self.avgpool1 = nn.AvgPool2d(7)
        # ofm = 1024,1,1

        self.fc1 = nn.Linear(1024, 200, bias=False)
        # ofm = 200

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.bn4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.bn5(x)

        x = self.conv6(x)
        x = self.relu6(x)
        x = self.bn6(x)

        x = self.conv7(x)
        x = self.relu7(x)
        x = self.bn7(x)

        x = self.conv8(x)
        x = self.relu8(x)
        x = self.bn8(x)

        x = self.conv9(x)
        x = self.relu9(x)
        x = self.bn9(x)

        x = self.conv10(x)
        x = self.relu10(x)
        x = self.bn10(x)

        x = self.conv11(x)
        x = self.relu11(x)
        x = self.bn11(x)

        x = self.avgpool1(x)

        x = x.view(-1, 1024)
        x = self.fc1(x)

        return x

'''
class MobileNet_trimmed3_spiking(nn.Module):
    def __init__(self, thresholds):
    #def __init__(self, thresholds, device):
        super(MobileNet_trimmed3_spiking, self).__init__()

        #size = 3,56,56 (stride=2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, groups=1, bias=False)
        self.relu1 = spikeRelu(thresholds[0])
        self.bn1 = nn.BatchNorm2d(32)
        self.bn1 = self.bn1.to(device)
        self.relu2 = spikeRelu(thresholds[1])
        #self.relu1 = spikeRelu(thresholds[0], monitor=True, index=(0,0,0,0))
        # ofm = 32,28,28

        # dw-conv-1 (stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32, bias=False)
        self.relu3 = spikeRelu(thresholds[2])
        self.bn2 = nn.BatchNorm2d(32)
        self.bn2 = self.bn2.to(device)
        self.relu4 = spikeRelu(thresholds[3])
        # ofm = 32,28,28
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.relu5 = spikeRelu(thresholds[4])
        self.bn3 = nn.BatchNorm2d(64)
        self.bn3 = self.bn3.to(device)
        self.relu6 = spikeRelu(thresholds[5])
        # ofm = 64,28,28

        # dw-conv-2 (stride=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, groups=64, bias=False)
        self.relu7 = spikeRelu(thresholds[6])
        self.bn4 = nn.BatchNorm2d(64)
        self.bn4 = self.bn4.to(device)
        self.relu8 = spikeRelu(thresholds[7])
        # ofm = 64,14,14
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.relu9 = spikeRelu(thresholds[8])
        self.bn5 = nn.BatchNorm2d(128)
        self.bn5 = self.bn5.to(device)
        self.relu10 = spikeRelu(thresholds[9])
        # ofm = 128,14,14

        # dw-conv-3 (stride=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128, bias=False)
        self.relu11 = spikeRelu(thresholds[10])
        self.bn6 = nn.BatchNorm2d(128)
        self.bn6 = self.bn6.to(device)
        self.relu12 = spikeRelu(thresholds[11])
        # ofm = 128,14,14
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.relu13 = spikeRelu(thresholds[12])
        self.bn7 = nn.BatchNorm2d(256)
        self.bn7 = self.bn7.to(device)
        self.relu14 = spikeRelu(thresholds[13])
        # ofm = 256,14,14

        # dw-conv-4 (stride=2)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, groups=256, bias=False)
        self.relu15 = spikeRelu(thresholds[14])
        self.bn8 = nn.BatchNorm2d(256)
        self.bn8 = self.bn8.to(device)
        self.relu16 = spikeRelu(thresholds[15])
        # ofm = 256,7,7
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.relu17 = spikeRelu(thresholds[16])
        self.bn9 = nn.BatchNorm2d(512)
        self.bn9 = self.bn9.to(device)
        self.relu18 = spikeRelu(thresholds[17])
        # ofm = 512,7,7

        # dw-conv-5 (stride=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=512, bias=False)
        self.relu19 = spikeRelu(thresholds[18])
        self.bn10 = nn.BatchNorm2d(512)
        self.bn10 = self.bn10.to(device)
        self.relu20 = spikeRelu(thresholds[19])
        # ofm = 512,7,7
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.relu21 = spikeRelu(thresholds[20])
        self.bn11 = nn.BatchNorm2d(1024)
        self.bn11 = self.bn11.to(device)
        self.relu22 = spikeRelu(thresholds[21])
        # ofm = 1024,7,7

        self.avgpool1 = nn.AvgPool2d(7)
        self.relu23 = spikeRelu(thresholds[22])
        # ofm = 1024,1,1

        self.fc1 = nn.Linear(1024, 200, bias=False)
        self.relu24 = spikeRelu(thresholds[23])
        # ofm = 200

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.relu2(x)

        x = self.conv2(x)
        x = self.relu3(x)
        x = self.bn2(x)
        x = self.relu4(x)

        x = self.conv3(x)
        x = self.relu5(x)
        x = self.bn3(x)
        x = self.relu6(x)

        x = self.conv4(x)
        x = self.relu7(x)
        x = self.bn4(x)
        x = self.relu8(x)

        x = self.conv5(x)
        x = self.relu9(x)
        x = self.bn5(x)
        x = self.relu10(x)

        x = self.conv6(x)
        x = self.relu11(x)
        x = self.bn6(x)
        x = self.relu12(x)

        x = self.conv7(x)
        x = self.relu13(x)
        x = self.bn7(x)
        x = self.relu14(x)

        x = self.conv8(x)
        x = self.relu15(x)
        x = self.bn8(x)
        x = self.relu16(x)

        x = self.conv9(x)
        x = self.relu17(x)
        x = self.bn9(x)
        x = self.relu18(x)

        x = self.conv10(x)
        x = self.relu19(x)
        x = self.bn10(x)
        x = self.relu20(x)

        x = self.conv11(x)
        x = self.relu21(x)
        x = self.bn11(x)
        x = self.relu22(x)

        x = self.avgpool1(x)
        x = self.relu23(x)

        x = x.view(-1, 1024)
        x = self.fc1(x)
        x = self.relu24(x)

        return x
'''

class MobileNet_trimmed3_spiking(nn.Module):
    def __init__(self, thresholds):
    #def __init__(self, thresholds, device):
        super(MobileNet_trimmed3_spiking, self).__init__()

        #size = 3,56,56 (stride=2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn1 = self.bn1.to(device)
        self.relu1 = spikeRelu(thresholds[0])
        #self.relu1 = spikeRelu(thresholds[0], monitor=True, index=(0,0,0,0))
        # ofm = 32,28,28

        # dw-conv-1 (stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn2 = self.bn2.to(device)
        self.relu2 = spikeRelu(thresholds[1])
        # ofm = 32,28,28
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn3 = self.bn3.to(device)
        self.relu3 = spikeRelu(thresholds[2], monitor=True, index=(0,0,0,0))
        #self.relu3 = spikeRelu(thresholds[2])
        # ofm = 64,28,28

        # dw-conv-2 (stride=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, groups=64, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn4 = self.bn4.to(device)
        self.relu4 = spikeRelu(thresholds[3])
        # ofm = 64,14,14
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn5 = self.bn5.to(device)
        self.relu5 = spikeRelu(thresholds[4])
        # ofm = 128,14,14

        # dw-conv-3 (stride=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128, bias=False)
        self.bn6 = nn.BatchNorm2d(128)
        self.bn6 = self.bn6.to(device)
        self.relu6 = spikeRelu(thresholds[5])
        # ofm = 128,14,14
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.bn7 = nn.BatchNorm2d(256)
        self.bn7 = self.bn7.to(device)
        self.relu7 = spikeRelu(thresholds[6])
        # ofm = 256,14,14

        # dw-conv-4 (stride=2)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, groups=256, bias=False)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn8 = self.bn8.to(device)
        self.relu8 = spikeRelu(thresholds[7])
        # ofm = 256,7,7
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.bn9 = nn.BatchNorm2d(512)
        self.bn9 = self.bn9.to(device)
        self.relu9 = spikeRelu(thresholds[8])
        # ofm = 512,7,7

        # dw-conv-5 (stride=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=512, bias=False)
        self.bn10 = nn.BatchNorm2d(512)
        self.bn10 = self.bn10.to(device)
        self.relu10 = spikeRelu(thresholds[9])
        # ofm = 512,7,7
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.bn11 = nn.BatchNorm2d(1024)
        self.bn11 = self.bn11.to(device)
        self.relu11 = spikeRelu(thresholds[10])
        # ofm = 1024,7,7

        self.avgpool1 = nn.AvgPool2d(7)
        self.relu12 = spikeRelu(thresholds[11])
        # ofm = 1024,1,1

        self.fc1 = nn.Linear(1024, 200, bias=False)
        self.relu13 = spikeRelu(thresholds[12])
        # ofm = 200

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.bn4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.bn5(x)

        x = self.conv6(x)
        x = self.relu6(x)
        x = self.bn6(x)

        x = self.conv7(x)
        x = self.relu7(x)
        x = self.bn7(x)

        x = self.conv8(x)
        x = self.relu8(x)
        x = self.bn8(x)

        x = self.conv9(x)
        x = self.relu9(x)
        x = self.bn9(x)

        x = self.conv10(x)
        x = self.relu10(x)
        x = self.bn10(x)

        x = self.conv11(x)
        x = self.relu11(x)
        x = self.bn11(x)

        x = self.avgpool1(x)
        x = self.relu12(x)

        x = x.view(-1, 1024)
        x = self.fc1(x)
        x = self.relu13(x)

        return x
