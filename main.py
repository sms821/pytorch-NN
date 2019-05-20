import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np
from mobilenet_trimmed1 import MobileNet_trimmed
from data_loaders import tinyimagenet_get_datasets
from train_routines import *
from checkpoint import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(data_dir, epochs, arch, model_path=None, resume=False, chkpt_file=None):
    batch_size = 64
    start_epoch = 0

    train_dataset, val_dataset = tinyimagenet_get_datasets(data_dir)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if torch.cuda.is_available():
        print('GPU device {} available'.format(torch.cuda.get_device_name(0)))
    else:
        print('GPU device not available')

    model = MobileNet_trimmed().to(device)
    print(model)

    # Loss fn and optimizer
    criterion = nn.CrossEntropyLoss()
    # TODO: add an lr scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    best_prec1 = 0
    if resume:
        chkpt = os.path.join(model_path, chkpt_file)
        model, arch, optimizer, start_epoch, best_prec1 = load_checkpoint(model, chkpt, optimizer, device)

    for epoch in range(start_epoch, start_epoch+epochs):
        # train for current epoch
        train(train_loader, model, criterion, optimizer, epoch, device)

        # validate
        avg_prec1 = validate(val_loader, model, criterion, epoch, device)

        if avg_prec1 > best_prec1:
            best_prec1 = avg_prec1

            # Checkpoint the model
            save_checkpoint(epoch, arch, model, best_prec1, optimizer, model_path)

if __name__ == '__main__':
    main('./datasets/tinyimagenet/tiny-imagenet-200', 10, 'mobilenet_trimmed1', './saved_models')

