from math import floor
import torch
import torch.nn as nn
from spiking import spikeRelu

class lenet5(nn.Module):

    def __init__(self):
        super(lenet5, self).__init__()

        # set-1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, padding=2, bias=False)
        self.bin1 = nn.ReLU()
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # set-2
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, padding=2, bias=False)
        self.bin2 = nn.ReLU()
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # set-3
        self.fc3 = nn.Linear(50*7*7, 500, bias=False)
        self.bin3 = nn.ReLU()

        # set-4
        self.fc4 = nn.Linear(500, 10, bias=False)

        # intermediate output
        self.conv_relu1 = torch.zeros(1)
        self.avg1 = torch.zeros(1)
        self.conv_relu2 = torch.zeros(1)
        self.avg2 = torch.zeros(1)
        self.fc_relu1 = torch.zeros(1)
        self.last_layer = torch.zeros(1)

        self.conv_relu1 = self.conv_relu1.cuda()
        self.avg1 = self.avg1.cuda()
        self.conv_relu2 = self.conv_relu2.cuda()
        self.avg2 = self.avg2.cuda()
        self.fc_relu1 = self.fc_relu1.cuda()
        self.last_layer = self.last_layer.cuda()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bin1(x)
        self.conv_relu1 = torch.max(torch.max(x), self.conv_relu1) # output of 1st conv-relu layer

        x = self.avg_pool1(x)
        self.avg1 = torch.max(torch.max(x), self.avg1) # output of 1st avg-pool layer

        x = self.conv2(x)
        x = self.bin2(x)
        self.conv_relu2 = torch.max(torch.max(x), self.conv_relu2) # output of 2nd conv-relu layer

        x = self.avg_pool2(x)
        self.avg2 = torch.max(torch.max(x), self.avg2) # output of 2nd avg-pool layer

        x = x.view(-1, 7*7*50)
        x = self.fc3(x)
        x = self.bin3(x)
        self.fc_relu1 = torch.max(torch.max(x), self.fc_relu1)

        x = self.fc4(x)
        self.last_layer = torch.max(torch.max(x), self.last_layer)

        return x

    def print_max_act(self):
        print ('max activations: ')
        print ('conv_relu1: {:f}, avg1: {:f}, conv_relu2: {:f}, avg2: {:f}, fc_relu1: {:f}, fc2: {:f}' \
                .format(torch.max(self.conv_relu1), torch.max(self.avg1), torch.max(self.conv_relu2), torch.max(self.avg2), \
                torch.max(self.fc_relu1), torch.max(self.last_layer)))


class lenet5_spiking(nn.Module):

    def __init__(self, thresholds):
        super(lenet5_spiking, self).__init__()

        # set-1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, padding=2, bias=False)
        #self.bin1 = spikeRelu(thresholds[0], monitor=True, index=(0,0,0,0))
        self.bin1 = spikeRelu(thresholds[0])
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_spike1 = spikeRelu(thresholds[1])

        # set-2
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, padding=2, bias=False)
        self.bin2 = spikeRelu(thresholds[2])
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_spike2 = spikeRelu(thresholds[3])

        # set-3
        self.fc3 = nn.Linear(50*7*7, 500, bias=False)
        self.bin3 = spikeRelu(thresholds[4])

        # set-4
        self.fc4 = nn.Linear(500, 10, bias=False)
        self.bin4 = spikeRelu(thresholds[5])


    def forward(self, x):
        x = self.conv1(x)
        x = self.bin1(x)

        x = self.avg_pool1(x)
        x = self.avg_spike1(x)

        x = self.conv2(x)
        x = self.bin2(x)

        x = self.avg_pool2(x)
        x = self.avg_spike2(x)

        x = x.view(-1, 7*7*50)
        x = self.fc3(x)
        x = self.bin3(x)

        x = self.fc4(x)
        x = self.bin4(x)

        return x

    def get_nr_spike_train(self):
        return self.bin1.get_spike_train()
