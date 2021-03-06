import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np
from mobilenet_trimmed1 import *
from mobilenet_trimmed2 import *
from mobilenet_trimmed3 import *
from mobilenet_trimmed4 import *
from mobilenet_trimmed5 import *
from mobilenet import MobileNet
from lenet5 import *
from data_loaders import *
from train_routines import *
from checkpoint import *
from utils import *
from spiking import simulate_spiking_model, poisson_spikes
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Separable CNN training using tiny imagenet and pytorch')
parser.add_argument("-d", "--dataset", default='tiny-imagenet-200', \
        choices=['tiny-imagenet-200', 'mnist', 'tiny-imagenet-100'], \
        help="dataset to train on (default: tiny-imagenet-200)")

parser.add_argument("--data-dir", default='./datasets/tinyimagenet/tiny-imagenet-200', \
        help='path to dataset (default: ./datasets/tinyimagenet/tiny-imagenet-200)')

parser.add_argument("--batch-size", default=64, type=int, \
        help='batch size (default: 64)')

parser.add_argument("--epochs", default=25, type=int, \
        help='total number of epochs to train for (default: 25)')

parser.add_argument("--arch", default='mobilenet_trimmed1', \
        choices=['mobilenet_trimmed1', 'mobilenet_trimmed2', 'mobilenet_trimmed3', \
        'lenet5','mobilenet_trimmed1_drop', 'mobilenet', 'mobilenet_trimmed4', \
        'mobilenet_trimmed5'], \
        help='model to train (default: mobilenet_trimmed3)')

parser.add_argument("--lr", default=0.01, type=float, \
        help='learning rate (default: 0.01)')

parser.add_argument("--chkpt-dir", default='saved_models', \
        help='directory for saving checkpoints (default: saved_models)')

parser.add_argument("--chkpt-file", default='checkpoint.pth.tar', \
        help='checkpoint file name (default: checkpoint.pth.tar)')

parser.add_argument("--resume", action='store_true', \
        help='resumes training from checkpoint (default: False)')

#parser.add_argument("--mode", default='train', \
#        choices=['train', 'quantize', 'validate'], \
#        help='(default: False)')

parser.add_argument("--quantize", action='store_true', \
        help='quantize weights of a pre-trained model (default: False)')

parser.add_argument("--I", type=int, default=2, \
        help='fixed point integer part')

parser.add_argument("--F", type=int, default=4, \
        help='fixed point fractional part')

parser.add_argument("--train", action='store_true', \
        help='train a model (default: False)')

parser.add_argument("--validate", action='store_true', \
        help='test a model (default: False)')

parser.add_argument("--compute-thresholds", action='store_true', \
        help='compute threshold of each layer in the model (default: False)')

parser.add_argument("--simulate-spiking", action='store_true', \
        help='simulate an SNN for the given model (default: False)')

parser.add_argument("--time-window", type=int, default=10, \
        help='SNN simulation time window')

parser.add_argument("--spike-study", action='store_true', \
        help='study poisson spikes (default: False)')

parser.add_argument("--num-samples", type=int, default=1, \
        help='num of experiments')

parser.add_argument("--num-classes", type=int, default=200, \
        help='num of image classes')


def main(args):
    batch_size = args.batch_size
    start_epoch = 0

    # prepare dataset
    train_dataset, val_dataset = None, None
    if args.dataset=='tiny-imagenet-200' or args.dataset=='tiny-imagenet-100':
        train_dataset, val_dataset = tinyimagenet_get_datasets(args.data_dir)
    elif args.dataset == 'mnist':
        train_dataset, val_dataset = mnist_get_datasets(args.data_dir)
    else:
        print('invalid dataset {}'.format(args.dataset))

    # load dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if torch.cuda.is_available():
        print('GPU device {} available'.format(torch.cuda.get_device_name(0)))
    else:
        print('GPU device not available')

    # instantiate the model class
    input_size = (3,56,56)
    model = None
    if args.arch == 'mobilenet_trimmed1':
        model = MobileNet_trimmed()
    elif args.arch == 'mobilenet_trimmed1_drop':
        model = MobileNet_trimmed_dropout()
    elif args.arch == 'mobilenet_trimmed2':
        model = MobileNet_trimmed2()
    elif args.arch == 'mobilenet_trimmed4':
        model = MobileNet_trimmed4(args.num_classes)
    elif args.arch == 'mobilenet_trimmed5':
        model = MobileNet_trimmed5(args.num_classes)
    elif args.arch == 'lenet5':
        model = lenet5()
        input_size = (1,28,28)
    elif args.arch == 'mobilenet':
        model = MobileNet()
        input_size = (3,64,64)
    else:
        model = MobileNet_trimmed3()

    # transfer to device of choice
    model = model.to(device)
    summary(model, input_size=input_size)
    #return

    # Loss fn and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=30)
    best_prec1, avg_prec1 = 0, 0

    chkpt_dir = os.path.join(args.chkpt_dir, args.arch)
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)

    if args.resume or args.validate or args.compute_thresholds or args.simulate_spiking:
        chkpt_file = os.path.join(chkpt_dir, args.chkpt_file)
        if not os.path.exists(chkpt_file):
            print('No checkpoint file {}'.format(chkpt_file))

        model, optimizer, start_epoch, best_prec1, arch = load_checkpoint(model, chkpt_file, optimizer, device)

    #### TRAIN ####
    if args.train:
        for epoch in range(start_epoch, start_epoch+args.epochs):
            #adjust_learning_rate(optimizer, epoch, args.lr)

            # train for current epoch
            train(train_loader, model, criterion, optimizer, epoch, device)

            # validate
            avg_prec1 = validate(val_loader, model, criterion, epoch, device)

            if avg_prec1[0] > best_prec1:
                best_prec1 = avg_prec1[0]

                # Checkpoint the model
                save_checkpoint(epoch, args.arch, model, best_prec1, optimizer, chkpt_dir)

            scheduler.step(avg_prec1[0])

        save_checkpoint(start_epoch+args.epochs, args.arch, model, avg_prec1[0], optimizer, chkpt_dir)

    #### QUANTIZE ####
    if args.quantize:
        quantize_model(model, chkpt_dir, args.chkpt_file, device, args.I, args.F)

    #### VALIDATE ####
    if args.validate:
        avg_prec1 = validate(val_loader, model, criterion, 0, device)

    #### COMPUTE THRESHOLDS ####
    if args.compute_thresholds:
        # register hooks on every layer
        ls = []
        for m in model.modules():
            #ls.append(m) # apply hooks to ALL layers
            if type(m) == torch.nn.ReLU or type(m) == torch.nn.Linear \
                or type(m) == torch.nn.AvgPool2d:
                ls.append(m)

        hookF = [Hook(layer) for layer in ls]
        print('number of forward hooks: {}'.format(len(hookF)))

        avg_prec1, max_acts = validate(val_loader, model, criterion, 0, device, args.compute_thresholds, hookF)

        for v in max_acts:
            print(v)

        thresholds = torch.zeros(len(max_acts)-1)
        for i in range(len(max_acts)-1):
            thresholds[i] = max_acts[i+1] / max_acts[i]
        fnm = 'thresholds'+args.arch+'.txt'
        np.savetxt(fnm, thresholds.cpu().numpy(), fmt='%.4f')

    #### SIMULATE SPIKING ####
    if args.simulate_spiking:
        fnm = 'thresholds'+args.arch+'.txt'
        thresholds = np.loadtxt(fnm)
        thresholds = thresholds
        thresholds = torch.from_numpy(thresholds).float()
        thresholds = thresholds.to(device)
        #print(thresholds.dtype)
        spiking = None
        if args.arch == 'mobilenet_trimmed1':
            spiking = MobileNet_trimmed_spiking(thresholds)
        elif args.arch == 'mobilenet_trimmed3':
            spiking = MobileNet_trimmed3_spiking(thresholds)
        elif args.arch == 'lenet5':
            spiking = lenet5_spiking(thresholds)

        print (spiking)

        spiking = convert_to_spiking(model, spiking)
        avg_prec1, max_acts = validate(val_loader, model, criterion, 0, device)
        avg_prec1, max_acts = validate(val_loader, spiking, criterion, 0, device)

        #simulate_spiking_model(args.arch, model, val_loader, criterion, device, \
        #        args.time_window, args.batch_size, args.num_classes, thresholds)

    #### SPIKE STUDY ####
    if args.spike_study:
        for i, (input, labels) in enumerate(val_loader):
            if i > 0:
                break
            #print(np.where(input.numpy() < 0))
            input, labels = input.to(device), labels.to(device)
            spike_train = torch.zeros(args.time_window)
            approx_pixel = torch.zeros(args.num_samples)
            for n in range(args.num_samples):
                for t in range(args.time_window):
                    # convert image pixels to spiking inputs
                    spikes = poisson_spikes(input, device)
                    index = (0,0,3,0)
                    spike_train[t] = spikes[index]

                print("spike train {}".format(spike_train))
                approx_pixel[n] = torch.max(input) * torch.sum(spike_train) / args.time_window
            print("original pixel value: {}".format(input[index]))
            print("spike train approx. value: {}".  format(approx_pixel.mean()))




if __name__ == '__main__':
    args = parser.parse_args()
    print('Parameters: {}'.format(args))
    main(args)
