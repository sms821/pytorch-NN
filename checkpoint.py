import os
import shutil
from errno import ENOENT
import logging
from numbers import Number
from tabulate import tabulate
import torch

def save_checkpoint(epoch, arch, model, best_prec1,optimizer=None, dir='.'):
    """Save a pytorch training checkpoint

    Args:
        epoch: current epoch number
        arch: name of the network architecture/topology
        model: a pytorch model
        optimizer: the optimizer used in the training session
        scheduler: the CompressionScheduler instance used for training, if any
        extras: optional dict with additional user-defined data to be saved in the checkpoint.
            Will be saved under the key 'extras'
        is_best: If true, will save a copy of the checkpoint with the suffix 'best'
        name: the name of the checkpoint file
        dir: directory in which to save the checkpoint
    """
    if not os.path.isdir(dir):
        raise IOError(ENOENT, 'Checkpoint directory does not exist at', os.path.abspath(dir))

    str_val = '%.3f'%best_prec1
    filename = 'checkpoint_' + arch + '_' + str(str_val) + 'pth.tar'
    fullpath = os.path.join(dir, filename)
    print("Saving checkpoint to: %s" % fullpath)

    checkpoint = {}
    checkpoint['epoch'] = epoch
    checkpoint['arch'] = arch
    checkpoint['state_dict'] = model.state_dict()
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        checkpoint['optimizer_type'] = type(optimizer)

    torch.save(checkpoint, fullpath)


def load_lean_checkpoint(model, chkpt_file, model_device=None):
    return load_checkpoint(model, chkpt_file, model_device=model_device,
                           lean_checkpoint=True)[0]


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


def load_checkpoint(model, chkpt_file, optimizer=None, model_device=None, *, lean_checkpoint=False):
    """Load a pytorch training checkpoint.

    Args:
        model: the pytorch model to which we will load the parameters
        chkpt_file: the checkpoint file
        lean_checkpoint: if set, read into model only 'state_dict' field
        optimizer: [deprecated argument]
        model_device [str]: if set, call model.to($model_device)
                This should be set to either 'cpu' or 'cuda'.
    :returns: updated model, compression_scheduler, optimizer, start_epoch
    """
    if not os.path.isfile(chkpt_file):
        raise IOError(ENOENT, 'Could not find a checkpoint file at', chkpt_file)

    print("=> loading checkpoint %s", chkpt_file)
    checkpoint = torch.load(chkpt_file, map_location=lambda storage, loc: storage)

    print('=> Checkpoint contents:\n{}\n'.format(get_contents_table(checkpoint)))
    if 'state_dict' not in checkpoint:
        raise ValueError("Checkpoint must contain the model parameters under the key 'state_dict'")

    #checkpoint_epoch = checkpoint.get('epoch', None)
    start_epoch = checkpoint['epoch']

    model.load_state_dict(checkpoint['state_dict'])
    if model_device is not None:
        model.to(model_device)

    best_prec1 = checkpoint['best_prec1']
    arch = checkpoint['arch']
    return (model, optimizer, start_epoch+1, best_prec1, arch)
