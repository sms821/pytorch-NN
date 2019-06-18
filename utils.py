#from mobilenet_trimmed1 import *
import numpy as np
import torch
import os
from checkpoint import *

'''
class IntermediateLayers(torch.nn.Module):
    def __init__(self, model):
        super(IntermediateLayers, self).__init__()
        features = list(list(model.modules())[:-1][0])
        self.features = torch.nn.ModuleList(features).eval()
        print(self.features)
        self.results = torch.zeros(len(self.features))
        print(len(self.results))

    def forward(self, x):
        for i, net in enumerate(self.features):
            x = net(x)
            self.results[i] = torch.max(self.results[i], x)
        #print (self.results[0])

    def print_max_acts(self):
        print('max activation values for every layer: '.format(len(self.results)))
'''

def convert_to_spiking(model, spiking):
    bn_params = []
    for layer in model.modules():
        #print (layer.state_dict().keys())
        if isinstance(layer, torch.nn.BatchNorm2d):
            bn_params.append(layer)
            #print('gb', layer.weight.shape, layer.bias.shape)
            #print('mv', layer.running_mean.shape, layer.running_var.shape)
            #print(layer.num_batches_tracked)

    all_wts = []
    for param in model.parameters():
        #print(param)
        wts = param.data
        if len(wts.size()) > 1:
            all_wts.append(wts)
    print('no of wt layers: {}'.format(len(all_wts)))

    b, l = 0, 0
    for layer in spiking.modules():
        #if isinstance(layer, torch.nn.BatchNorm2d):
        #    layer = bn_params[b]
        #    b += 1
        #    print(layer.weight.shape, layer.bias.shape)
        #    print(layer.running_mean.shape, layer.running_var.shape)

        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
            print(torch.min(all_wts[l]), torch.max(all_wts[l]))
            layer.weight.data = all_wts[l]
            l += 1
    return spiking


class Hook():
    def __init__(self, module, backward=False):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


def copy_params(model_mod, model, device):
    # store weights in a list
    all_wts = []
    for param in model.parameters():
        #print (param.data.size())
        wts = param.data.cpu().numpy()
        if len(wts.shape) > 1:
            #print('wt max {}\twt min{}'.format(np.amax(wts), np.amin(wts)))
            all_wts.append(wts)
    print ('size of all_wts: {}'.format(len(all_wts)))
    l = 0
    for layer in model_mod.children():
        #print ('layer {} {}'.format(layer, layer.weight.data.size()))
        #print(l, all_wts[l].shape)
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
            temp = torch.from_numpy(all_wts[l])
            temp = temp.to(device)
            layer.weight.data = temp
            l+=1

    #for param in model_mod.parameters():
    #    print (param.data.size())
    #    wts = param.data.cpu().numpy()
    #    if len(wts.shape) > 1:
    #        print('wt max {}\twt min{}'.format(np.amax(wts), np.amin(wts)))

    print ('layers: {}'.format(l))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    quant_model_path = 'saved_models/mobilenet_trimmed1'
    filenm = 'mobnet_copy.pth.tar'
    save_checkpoint(0, 'mobilenet_trimmed1', model_mod, 68.750, optimizer, quant_model_path, filenm)
    return model_mod

def float_to_fixed(val, W=2, F=6):
    ''' converts floating point number val to fixed format W.F
    using method described in http://ee.sharif.edu/~asic/Tutorials/Fixed-Point.pdf '''
    nearest_int = round(val*(2**F))
    return nearest_int*(1.0/2**F)


def quantize_model(model, quant_model_path, chkpt_file, device, W=2, F=6):
    ''' Quantize the weights of a model

    Args:
        model: a pytorch model class
        quant_model_path: dir to load the original model and save the quantized model
        chkpt_file: checkpoint file from which to load the `model` params
        device: cpu, cuda
        W: num bits for integer part
        F: num bits for fractional part
    '''

    quantized_model = model
    quantized_model.to(device)

    model, optimizer, epoch, best_prec1, arch = \
            load_checkpoint(model, os.path.join(quant_model_path, chkpt_file))
    model = model.to(device)
    model.eval()
    print('Original model: ')
    print(model)

    quant_fn_vec = np.vectorize(float_to_fixed)

    # quantize weights and store them in a list
    all_wts = []
    for param in model.parameters():
        wts = param.data.cpu().numpy()
        print (wts.shape)
        if len(wts.shape) > 1:
            print('wt max {}\twt min{}'.format(np.amax(wts), np.amin(wts)))
            all_wts.append(quant_fn_vec(wts, W, F))
    print('no of wt layers: {}'.format(len(all_wts)))

    l = 0
    for layer in quantized_model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            print(all_wts[l].shape)
            temp = torch.from_numpy(all_wts[l])
            layer.weight.data = temp.to(device)
            l += 1
    print('Quantized model: ')
    print(quantized_model)

    print('Quantized model, saving the model to {}'.format(quant_model_path))
    filenm = arch + '_' + str(W) + "." + str(F) + 'quant.pth.tar'
    save_checkpoint(epoch, arch, quantized_model, best_prec1, optimizer, quant_model_path, filenm)

