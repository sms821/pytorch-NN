from math import floor
import torch.nn as nn

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
