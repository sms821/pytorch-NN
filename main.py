import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np
from mobilenet_trimmed1 import *
from data_loaders import tinyimagenet_get_datasets
from train_routines import *
from checkpoint import *
from utils import quantize_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Separable CNN training using tiny imagenet and pytorch')
parser.add_argument("-d", "--dataset", default='tiny-imagenet-200', \
        choices=['tiny-imagenet-200'], \
        help="dataset to train on (default: tiny-imagenet-200)")

parser.add_argument("--data-dir", default='./datasets/tinyimagenet/tiny-imagenet-200', \
        help='path to dataset (default: ./datasets/tinyimagenet/tiny-imagenet-200)')

parser.add_argument("--batch-size", default=64, type=int, \
        help='batch size (default: 64)')

parser.add_argument("--epochs", default=25, type=int, \
        help='total number of epochs to train for (default: 25)')

parser.add_argument("--arch", default='mobilenet_trimmed1', \
        choices=['mobilenet_trimmed1', 'mobilenet_trimmed1_drop'], \
        help='model to train (default: mobilenet_trimmed1)')

parser.add_argument("--lr", default=0.01, type=float, \
        help='learning rate (default: 0.01)')

parser.add_argument('--topk', default=1, type=int, \
        help='top-k accuracy (default: 1)')

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


def main(args):
    batch_size = args.batch_size
    start_epoch = 0

    # prepare dataset
    train_dataset, val_dataset = None, None
    if args.dataset == 'tiny-imagenet-200':
        train_dataset, val_dataset = tinyimagenet_get_datasets(args.data_dir)

    # load dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if torch.cuda.is_available():
        print('GPU device {} available'.format(torch.cuda.get_device_name(0)))
    else:
        print('GPU device not available')

    # instantiate the model class
    model = None
    if args.arch == 'mobilenet_trimmed1':
        model = MobileNet_trimmed()
    elif args.arch == 'mobilenet_trimmed1_drop':
        model = MobileNet_trimmed_dropout()

    # transfer to device of choice
    if args.train or args.validate:
        model = model.to(device)
    print(model)

    # Loss fn and optimizer
    criterion = nn.CrossEntropyLoss()
    # TODO: add an lr scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    best_prec1, avg_prec1 = 0, 0

    chkpt_dir = os.path.join(args.chkpt_dir, args.arch)
    if args.resume or args.validate:
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)

        chkpt_file = os.path.join(chkpt_dir, args.chkpt_file)
        if not os.path.exists(chkpt_file):
            print('No checkpoint file {}'.format(chkpt_file))

        #chkpt = os.path.join(model_path, chkpt_file)
        model, optimizer, start_epoch, best_prec1, arch = load_checkpoint(model, chkpt_file, optimizer, device)

    if args.train:
        for epoch in range(start_epoch, start_epoch+args.epochs):
            # train for current epoch
            train(train_loader, model, criterion, optimizer, epoch, device)

            # validate
            avg_prec1 = validate(val_loader, model, criterion, epoch, device)

            if avg_prec1 > best_prec1:
                best_prec1 = avg_prec1

                # Checkpoint the model
                save_checkpoint(epoch, args.arch, model, best_prec1, optimizer, chkpt_dir)
        save_checkpoint(start_epoch+args.epochs, args.arch, model, avg_prec1, optimizer, chkpt_dir)

    if args.quantize:
        quantize_model(model, chkpt_dir, args.chkpt_file, device, args.I, args.F)

    if args.validate:
        # validate
        #model.load_state_dict(torch.load(model_path))
        avg_prec1 = validate(val_loader, model, criterion, 0, device)


if __name__ == '__main__':
    args = parser.parse_args()
    print('Parameters: {}'.format(args))
    main(args)
