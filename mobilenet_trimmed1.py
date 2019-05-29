from math import floor
import torch
import torch.nn as nn
#from spiking import SpikeRelu
from spiking import spikeRelu

class MobileNet_trimmed(nn.Module):
    def __init__(self, channel_multiplier=1.0, min_channels=8):
        super(MobileNet_trimmed, self).__init__()

        if channel_multiplier <= 0:
            raise ValueError('channel_multiplier must be >= 0')

        def conv_bn_relu(n_ifm, n_ofm, kernel_size, stride=1, padding=0, groups=1):
            return [
                nn.Conv2d(n_ifm, n_ofm, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
                nn.BatchNorm2d(n_ofm),
                nn.ReLU(inplace=True)
            ]

        def depthwise_conv(n_ifm, n_ofm, stride):
            return nn.Sequential(
                *conv_bn_relu(n_ifm, n_ifm, 3, stride=stride, padding=1, groups=n_ifm),
                *conv_bn_relu(n_ifm, n_ofm, 1, stride=1)
            )

        base_channels = [32, 64, 128, 256]
        self.channels = [max(floor(n * channel_multiplier), min_channels) for n in base_channels]

        self.model = nn.Sequential(
            nn.Sequential(*conv_bn_relu(3, self.channels[0], 3, stride=1, padding=1)),
            # ofm: 32,64,64
            nn.AvgPool2d(2),
            # ofm: 32,32,32

            depthwise_conv(self.channels[0], self.channels[1], 1),
            # ofm: 64,32,32
            nn.AvgPool2d(2),
            # ofm: 64,16,16

            depthwise_conv(self.channels[1], self.channels[2], 1),
            # ofm: 128,16,16
            nn.AvgPool2d(2),
            # ofm: 128,8,8

            depthwise_conv(self.channels[2], self.channels[3], 2),
            # ofm: 256,8,8
            nn.AvgPool2d(8),
            # ofm: 256,1,1
        )
        self.fc = nn.Linear(self.channels[3], 200, bias=False)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, self.channels[-1])
        x = self.fc(x)
        return x

class MobileNet_trimmed_spiking(nn.Module):
    def __init__(self, thresholds, channel_multiplier=1.0, min_channels=8):
        super(MobileNet_trimmed_spiking, self).__init__()

        if channel_multiplier <= 0:
            raise ValueError('channel_multiplier must be >= 0')

        def conv_bn_relu(n_ifm, n_ofm, kernel_size, th_idx, stride=1, padding=0, groups=1):
            return [
                nn.Conv2d(n_ifm, n_ofm, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
                #nn.BatchNorm2d(n_ofm),
                #nn.ReLU(inplace=True)
                #SpikeRelu(thresholds[th_idx])
                spikeRelu(thresholds[th_idx])
            ]

        def depthwise_conv(n_ifm, n_ofm, stride, idx):
            return nn.Sequential(
                *conv_bn_relu(n_ifm, n_ifm, 3, idx, stride=stride, padding=1, groups=n_ifm),
                *conv_bn_relu(n_ifm, n_ofm, 1, idx+1, stride=1)
            )

        base_channels = [32, 64, 128, 256]
        self.channels = [max(floor(n * channel_multiplier), min_channels) for n in base_channels]

        self.model = nn.Sequential(
            nn.Sequential(*conv_bn_relu(3, self.channels[0], 3, th_idx=0, stride=1, padding=1)),
            # ofm: 32,64,64
            nn.AvgPool2d(2),
            # ofm: 32,32,32
            #SpikeRelu(thresholds[1]),
            spikeRelu(thresholds[1]),

            depthwise_conv(self.channels[0], self.channels[1], 1, idx=2),
            # ofm: 64,32,32
            nn.AvgPool2d(2),
            # ofm: 64,16,16
            #SpikeRelu(thresholds[4]),
            spikeRelu(thresholds[4]),

            depthwise_conv(self.channels[1], self.channels[2], 1, idx=5),
            # ofm: 128,16,16
            nn.AvgPool2d(2),
            # ofm: 128,8,8
            #SpikeRelu(thresholds[7]),
            spikeRelu(thresholds[7]),

            depthwise_conv(self.channels[2], self.channels[3], 2, idx=8),
            # ofm: 256,8,8
            nn.AvgPool2d(8),
            # ofm: 256,1,1
            #SpikeRelu(thresholds[10])
            spikeRelu(thresholds[10])
        )
        self.fc = nn.Linear(self.channels[3], 200, bias=False)
        #self.last = SpikeRelu(thresholds[11])
        self.last = spikeRelu(thresholds[11])

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, self.channels[-1])
        x = self.fc(x)
        x = self.last(x)
        return x

class MobileNet_trimmed_copy(nn.Module):
    def __init__(self, device, channel_multiplier=1.0, min_channels=8):
        super(MobileNet_trimmed_copy, self).__init__()

        self.device = device
        if channel_multiplier <= 0:
            raise ValueError('channel_multiplier must be >= 0')

        def conv_bn_relu(n_ifm, n_ofm, kernel_size, stride=1, padding=0, groups=1):
            return [
                nn.Conv2d(n_ifm, n_ofm, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
                nn.BatchNorm2d(n_ofm),
                nn.ReLU(inplace=True)
            ]

        def depthwise_conv(n_ifm, n_ofm, stride):
            return nn.Sequential(
                *conv_bn_relu(n_ifm, n_ifm, 3, stride=stride, padding=1, groups=n_ifm),
                *conv_bn_relu(n_ifm, n_ofm, 1, stride=1)
            )

        base_channels = [32, 64, 128, 256]
        self.channels = [max(floor(n * channel_multiplier), min_channels) for n in base_channels]

        self.conv1 = nn.Conv2d(3, self.channels[0], 3, stride=1, padding=1, groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.channels[0])
        self.relu1 = nn.ReLU(inplace=True)
        # ofm: 32,64,64

        self.avg_pool1 = nn.AvgPool2d(2)
        # ofm: 32,32,32

        # dw1
        self.conv2 = nn.Conv2d(self.channels[0], self.channels[0], 3, stride=1, padding=1, groups=self.channels[0], bias=False)
        self.bn2 = nn.BatchNorm2d(self.channels[0])
        self.relu2 = nn.ReLU(inplace=True)
        # ofm: 32,32,32
        self.conv3 = nn.Conv2d(self.channels[0], self.channels[1], 1, stride=1, padding=0, groups=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.channels[1])
        self.relu3 = nn.ReLU(inplace=True)
        # ofm: 64,32,32

        self.avg_pool2 = nn.AvgPool2d(2)
        # ofm: 64,16,16

        # dw2
        self.conv4 = nn.Conv2d(self.channels[1], self.channels[1], 3, stride=1, padding=1, groups=self.channels[1], bias=False)
        self.bn4 = nn.BatchNorm2d(self.channels[1])
        self.relu4 = nn.ReLU(inplace=True)
        # ofm: 64,16,16
        self.conv5 = nn.Conv2d(self.channels[1], self.channels[2], 1, stride=1, padding=0, groups=1, bias=False)
        self.bn5 = nn.BatchNorm2d(self.channels[2])
        self.relu5 = nn.ReLU(inplace=True)
        # ofm: 128,16,16

        self.avg_pool3 = nn.AvgPool2d(2)
        # ofm: 128,8,8

        # dw3
        self.conv6 = nn.Conv2d(self.channels[2], self.channels[2], 3, stride=1, padding=1, groups=self.channels[2], bias=False)
        self.bn6 = nn.BatchNorm2d(self.channels[2])
        self.relu6 = nn.ReLU(inplace=True)
        # ofm: 128,8,8
        self.conv7 = nn.Conv2d(self.channels[2], self.channels[3], 1, stride=1, padding=0, groups=1, bias=False)
        self.bn7 = nn.BatchNorm2d(self.channels[3])
        self.relu7 = nn.ReLU(inplace=True)
        # ofm: 256,8,8

        self.avg_pool4 = nn.AvgPool2d(8)
        # ofm: 256,1,1

        self.fc = nn.Linear(self.channels[3], 200, bias=False)
        # 256,200

        # intermediate layer max activations
        device = self.device
        #self.conv1 = self.conv1.to(device)
        #self.bn1 = self.bn1.to(device)
        #self.relu1 = self.relu1.to(device)
        #self.avg_pool1 = self.avg_pool1.to(device)

        #self.conv2 = self.conv2.to(device)
        #self.bn2 = self.bn2.to(device)
        #self.relu2 = self.relu2.to(device)
        #self.avg_pool2 = self.avg_pool2.to(device)

        #self.conv3 = self.conv3.to(device)
        #self.bn3 = self.bn3.to(device)
        #self.relu3 = self.relu3.to(device)
        #self.avg_pool3 = self.avg_pool3.to(device)

        #self.conv4 = self.conv4.to(device)
        #self.bn4 = self.bn4.to(device)
        #self.relu4 = self.relu4.to(device)
        #self.avg_pool4 = self.avg_pool4.to(device)

        #self.conv5 = self.conv5.to(device)
        #self.bn5 = self.bn5.to(device)
        #self.relu5 = self.relu5.to(device)

        #self.conv6 = self.conv6.to(device)
        #self.bn6 = self.bn6.to(device)
        #self.relu6 = self.relu6.to(device)

        #self.conv7 = self.conv7.to(device)
        #self.bn7 = self.bn7.to(device)
        #self.relu7 = self.relu7.to(device)

        #self.fc = self.fc.to(device)

        self.conv_relu1, self.conv_relu2 = torch.zeros(1).to(device), torch.zeros(1).to(device)
        self.conv_relu3, self.conv_relu4 = torch.zeros(1).to(device), torch.zeros(1).to(device)
        self.conv_relu5, self.conv_relu6 = torch.zeros(1).to(device), torch.zeros(1).to(device)
        self.conv_relu7 = torch.zeros(1).to(device)
        self.avg1, self.avg2 = torch.zeros(1).to(device), torch.zeros(1).to(device)
        self.avg3, self.avg4 = torch.zeros(1).to(device), torch.zeros(1).to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        self.conv_relu1 = torch.max(self.conv_relu1, x)

        x = self.avg_pool1(x)
        self.avg1 = torch.max(self.avg1, x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        self.conv_relu2 = torch.max(self.conv_relu2, x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        self.conv_relu3 = torch.max(self.conv_relu3, x)

        x = self.avg_pool2(x)
        self.avg2 = torch.max(self.avg2, x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        self.conv_relu4 = torch.max(self.conv_relu4, x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        self.conv_relu5 = torch.max(self.conv_relu5, x)

        x = self.avg_pool3(x)
        self.avg3 = torch.max(self.avg3, x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        self.conv_relu6 = torch.max(self.conv_relu6, x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu7(x)
        self.conv_relu7 = torch.max(self.conv_relu7, x)

        x = self.avg_pool4(x)
        self.avg4 = torch.max(self.avg4, x)

        x = x.view(-1, self.channels[-1])
        x = self.fc(x)
        return x

    def print_max_acts(self):
        print('max activation values: ')
        print('conv_relu1: {}'.format(torch.max(self.conv_relu5)))

class MobileNet_trimmed_dropout(nn.Module):
    def __init__(self, channel_multiplier=1.0, min_channels=8):
        super(MobileNet_trimmed_dropout, self).__init__()

        if channel_multiplier <= 0:
            raise ValueError('channel_multiplier must be >= 0')

        def conv_bn_relu(n_ifm, n_ofm, kernel_size, stride=1, padding=0, groups=1):
            return [
                nn.Conv2d(n_ifm, n_ofm, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
                nn.BatchNorm2d(n_ofm),
                nn.ReLU(inplace=True)
            ]

        def depthwise_conv(n_ifm, n_ofm, stride):
            return nn.Sequential(
                *conv_bn_relu(n_ifm, n_ifm, 3, stride=stride, padding=1, groups=n_ifm),
                *conv_bn_relu(n_ifm, n_ofm, 1, stride=1)
            )

        base_channels = [32, 64, 128, 256]
        self.channels = [max(floor(n * channel_multiplier), min_channels) for n in base_channels]

        self.model = nn.Sequential(
            nn.Sequential(*conv_bn_relu(3, self.channels[0], 3, stride=1, padding=1)),
            # ofm: 32,64,64
            nn.AvgPool2d(2),
            # ofm: 32,32,32

            depthwise_conv(self.channels[0], self.channels[1], 1),
            # ofm: 64,32,32
            nn.AvgPool2d(2),
            # ofm: 64,16,16

            depthwise_conv(self.channels[1], self.channels[2], 1),
            # ofm: 128,16,16
            nn.AvgPool2d(2),
            # ofm: 128,8,8

            depthwise_conv(self.channels[2], self.channels[3], 1),
            # ofm: 256,8,8
            nn.AvgPool2d(8),
            # ofm: 256,1,1

        )
        self.fc = nn.Linear(self.channels[3], 200, bias=False)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, self.channels[-1])
        x = self.fc(x)
        return x
