import os
import shutil
from errno import ENOENT
import logging
from numbers import Number
from tabulate import tabulate
import torch

def save_checkpoint(epoch, arch, model, best_prec1, optimizer=None, dir='.', filename=None):
    """Save a pytorch training checkpoint

    Args:
        epoch: current epoch number
        arch: name of the network architecture/topology
        model: a pytorch model
        best_prec1: top1 acc of the model
        optimizer: the optimizer used in the training session
        dir: directory in which to save the checkpoint
        filename: the name of the checkpoint file
    """
    if not os.path.isdir(dir):
        raise IOError(ENOENT, 'Checkpoint directory does not exist at', os.path.abspath(dir))

    print(best_prec1)
    str_val = '%.3f'%best_prec1
    if not filename:
        filename = 'checkpoint_' + str(str_val) + '.pth.tar'
    fullpath = os.path.join(dir, filename)
    print("Saving checkpoint to: {}".format(fullpath))

    checkpoint = {}
    checkpoint['epoch'] = epoch
    checkpoint['arch'] = arch
    checkpoint['best_prec1'] = best_prec1
    checkpoint['state_dict'] = model.state_dict()
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        checkpoint['optimizer_type'] = type(optimizer)

    torch.save(checkpoint, fullpath)


def get_contents_table(d):
    def inspect_val(val):
        if isinstance(val, (Number, str)):
            return val
        elif isinstance(val, type):
            return val.__name__
        return None

    contents = [[k, type(d[k]).__name__, inspect_val(d[k])] for k in d.keys()]
    contents = sorted(contents, key=lambda entry: entry[0])
    return tabulate(contents, headers=["Key", "Type", "Value"], tablefmt="fancy_grid")


def load_checkpoint(model, chkpt_file, optimizer=None, model_device=None):
    """Load a pytorch training checkpoint.

    Args:
        model: the pytorch model to which we will load the parameters
        chkpt_file: the checkpoint file
        optimizer: [deprecated argument]
        model_device [str]: if set, call model.to($model_device)
                This should be set to either 'cpu' or 'cuda'.
    :returns: updated model, optimizer, start_epoch, best_prec1, arch
    """
    if not os.path.isfile(chkpt_file):
        raise IOError(ENOENT, 'Could not find a checkpoint file at', chkpt_file)

    print("=> loading checkpoint %s", chkpt_file)
    checkpoint = torch.load(chkpt_file, map_location=lambda storage, loc: storage)
    #checkpoint = torch.load(chkpt_file)
    print('checkpoint contents'.format(checkpoint.keys()))

    print('=> Checkpoint contents:\n{}\n'.format(get_contents_table(checkpoint)))
    #if 'state_dict' not in checkpoint:
    #    raise ValueError("Checkpoint must contain the model parameters under the key 'state_dict'")

    if 'epoch' in checkpoint:
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    #model.module.load_state_dict(checkpoint['state_dict'])
    #model.load_state_dict(checkpoint['state_dict'], strict=False)
    if 'state_dict' not in checkpoint:
        model.load_state_dict(checkpoint, strict=False)
    else:
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    if model_device is not None:
        model.to(model_device)

    best_prec1 = 0
    if 'best_prec1' in checkpoint:
        best_prec1 = checkpoint['best_prec1']

    arch = ''
    if 'arch' in checkpoint:
        arch = checkpoint['arch']
    return (model, optimizer, start_epoch+1, best_prec1, arch)
