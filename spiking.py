import numpy as np
import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import datetime
from train_routines import AverageMeter, accuracy
from utils import convert_to_spiking
from utils import Hook
import time

#def SpikeRelu(v_th):
#    class spikeRelu(nn.Module):
#
#        def __init__(self, v_th):
#            super(spikeRelu, self).__init__()
#            self.threshold = v_th
#            self.vmem = 0
#            self.time = 0
#            print(self.threshold)
#
#        def forward(self, x):
#            self.vmem += x
#
#            '''
#            print('simulation time: ', self.time)
#
#            print('input value added: ', x.size())
#            file1_nm = 'nrInput' + datetime.datetime.now().strftime("%H-%M-%S-%f")+'__tick-'+str(self.time)+'.txt'
#            np.savetxt(file1_nm, x.flatten().cpu().detach().numpy(), fmt='%d')
#
#            print('after adding spikes: ', self.vmem.size())
#            file2_nm = 'nrPotential' + datetime.datetime.now().strftime("%H-%M-%S-%f")+'__tick-'+str(self.time)+'.txt'
#            np.savetxt(file2_nm, self.vmem.flatten().detach().numpy(), fmt='%d')
#            '''
#
#            #op_spikes = torch.where(self.vmem.to('cuda:0') >= self.threshold, torch.ones(1).to('cuda:0'), torch.zeros(1).to('cuda:0'))
#            #ones = torch.ones(1)
#            #zeros = torch.zeros(1)
#            op_spikes = torch.where(self.vmem >= self.threshold, torch.ones(1), torch.zeros(1))
#            self.vmem = torch.where(self.vmem >= self.threshold, torch.zeros(1), self.vmem)
#            #self.vmem = torch.where(self.vmem.to('cuda:0') >= self.threshold, torch.zeros(1).to('cuda:0'), self.vmem.to('cuda:0'))
#
#            self.time += 1
#            return op_spikes
#
#        def extra_repr(self):
#            return 'v_th : {}, vmem : {}'.format(v_th, self.vmem)
#
#    return spikeRelu(v_th)


class spikeRelu(nn.Module):

    def __init__(self, v_th):
        super(spikeRelu, self).__init__()
        self.threshold = v_th
        self.vmem = 0
        self.time = 0
        #print(self.threshold)

    def forward(self, x):
        self.vmem += x
        idx = None
        if len(self.vmem.size()) > 2:
            idx = (0,0,0,0)
        else:
            idx = (0,0)
        #print(self.vmem[idx])

        '''
        print('simulation time: ', self.time)

        print('input value added: ', x.size())
        file1_nm = 'nrInput' + datetime.datetime.now().strftime("%H-%M-%S-%f")+'__tick-'+str(self.time)+'.txt'
        np.savetxt(file1_nm, x.flatten().cpu().detach().numpy(), fmt='%d')

        print('after adding spikes: ', self.vmem.size())
        file2_nm = 'nrPotential' + datetime.datetime.now().strftime("%H-%M-%S-%f")+'__tick-'+str(self.time)+'.txt'
        np.savetxt(file2_nm, self.vmem.flatten().detach().numpy(), fmt='%d')
        '''

        #op_spikes = torch.where(self.vmem.to('cuda:0') >= self.threshold, torch.ones(1).to('cuda:0'), torch.zeros(1).to('cuda:0'))
        #ones = torch.ones(1)
        #zeros = torch.zeros(1)
        op_spikes = torch.where(self.vmem >= self.threshold, torch.ones(1), torch.zeros(1))
        self.vmem = torch.where(self.vmem >= self.threshold, torch.zeros(1), self.vmem)
        #self.vmem = torch.where(self.vmem.to('cuda:0') >= self.threshold, torch.zeros(1).to('cuda:0'), self.vmem.to('cuda:0'))

        self.time += 1
        return op_spikes

    def extra_repr(self):
        return 'v_th : {}, vmem : {}'.format(self.threshold, self.vmem)



def condition(rand_val, in_val, abs_max_val):
    if rand_val <= (abs(in_val) / abs_max_val):
        return (np.sign(in_val))
    else:
        return 0

############### USE TORCH.WHERE!!!!!!!!!!!!!!11
def poisson_spikes(pixel_vals, device):
    #def max_axis(tensor, axes=None):
    #    if axes:
    #        temp = (tensor, 0)
    #        for a in sorted(axes, reverse=True):
    #            temp = torch.max(temp[0], dim=a)
    #        return temp[0]
    #    else:
    #        return torch.max(tensor)
    #random_inputs = torch.rand(pixel_vals.size())
    #out_spikes = torch.where(random_inputs <= (torch.abs(pixel_vals)))
    #out_spikes = np.zeros(pixel_vals.shape)
    out_spikes = torch.zeros(pixel_vals.size())
    out_spikes = out_spikes.to(device)
    for b in range(pixel_vals.size()[0]):
        random_inputs = torch.rand(pixel_vals.size()[1],pixel_vals.size()[2], pixel_vals.size()[3]).to(device)
        single_img = pixel_vals[b,:,:,:]
        max_val = torch.max(single_img)
        out_spikes[b,:,:,:] = torch.where(random_inputs <= (torch.abs(single_img)) / max_val, \
                torch.sign(single_img), torch.zeros(1).to(device))
    return out_spikes




def simulate_spiking_model(model, spike_model, val_loader, criterion, device, time_window, batch_size):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    num_classes = 200

    # switch to evaluate mode
    spike_model.eval()
    #print(model)

    end = time.time()

    total_correct = 0
    total_images = 0
    confusion_matrix = np.zeros([num_classes,num_classes], int)
    out_spikes_t_b_c = torch.zeros((time_window, batch_size, num_classes))

    for i, (input, labels) in enumerate(val_loader):
        print ('batch number: {}'.format(i))
        #if i > 100:
        #    break
        input, labels = input.to(device), labels.to(device)
        #print("labels: {}".format(labels.size()))
        spike_model = convert_to_spiking(model, spike_model)
        spike_model.eval()
        # creating a forward hook here for debugging
        ls = []
        for m in spike_model.modules():
            #if isinstance(m, torch.nn.Conv2d):
            #if isinstance(m, torch.nn.BatchNorm2d):
            if isinstance(m, spikeRelu):
                ls.append(m)
                break
        hookF = [Hook(layer) for layer in ls]

        for t in range(time_window):
            # convert image pixels to spiking inputs
            spikes = poisson_spikes(input, device)

            with torch.no_grad():
                out_spikes = spike_model(spikes)

            #for j, hook in enumerate(hookF):
            #    print(hook.output)
                #print(hook.input)

                out_spikes_t_b_c[t,:,:] = out_spikes

        # accumulating output spikes for all images in a batch
        total_spikes_b_c = torch.zeros((batch_size, num_classes))
        for b in range(batch_size):
            total_spikes_per_input = torch.zeros((num_classes))
            for t in range(time_window):
                total_spikes_per_input += out_spikes_t_b_c[t,b,:]
            #print ("total spikes per output: {}".format(total_spikes_per_input / time_window))
            #print ("total spikes per output: {}".format(total_spikes_per_input ))
            total_spikes_b_c[b,:] = total_spikes_per_input
            #total_spikes_b_c[b,:] = total_spikes_per_input / time_window # note the change

        loss = criterion(total_spikes_b_c, labels)

        # measure accuracy and record loss
        prec1 = accuracy(total_spikes_b_c.data, labels, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 25 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
        #_, predicted = torch.max(total_spikes_b_c.data, 1)
        #print("predicted: {}".format(predicted))
        ##print(predicted, labels)
        #total_images += labels.size(0)
        #total_correct += (predicted == labels).sum().item()
        #for i, l in enumerate(labels):
        #    confusion_matrix[l.item(), predicted[i].item()] += 1

    #model_accuracy = total_correct / total_images * 100
    #print('Model accuracy on {0} test images: {1:.2f}%'.format(total_images, model_accuracy))



'''
def main():
    model_path = 'data/spiking_bnn_checkpoints_actual_errors'
    fileNm = 'spiking_bnn_94.220.pth'
    #model_path = 'data/spiking_float'
    #fileNm = 'spiking_bnn_96.160.pth'
    #dummy_input = 2*np.random.rand(784) - np.ones(784)
    #print ('shape of input: {}. max value of input: {}. min value of input: {}'.format(dummy_input.shape, np.amax(dummy_input), np.amin(dummy_input)))
    #time_window = 2
    #poisson_spikes(time_window, dummy_input)

    batch_size = 50
    #computeThresholds(model_path, fileNm, batch_size)

    simulateSpikeModel(model_path, fileNm)

        simulate
if __name__ == '__main__':
    main()
'''
'''
def extractNeuronalActivity(spikeModel):
    model = spikeModel
    #temp = model._modules.get('layer_stack')
    temp = model
    l2_out = temp._modules.get('2') # Save output of layer-2
    l5_out = temp._modules.get('5') # Save output of layer-5
    l8_out = temp._modules.get('8') # Save output of layer-7

    l2_out_copy = torch.zeros((batch_size, 600)) # copy of l2 out
    l5_out_copy = torch.zeros((batch_size, 600)) # copy of l5 out
    l8_out_copy = torch.zeros((batch_size, 10)) # copy of l7 out

    def get_ftrs_l2(m, i, o): # function to copy output of l2
        l2_out_copy.copy_(o.data)

    def get_ftrs_l5(m, i, o): # function to copy output of l5
        l5_out_copy.copy_(o.data)

    def get_ftrs_l8(m, i, o): # function to copy output of l7
        l8_out_copy.copy_(o.data)

    h2 = l2_out.register_forward_hook(get_ftrs_l2)
    h5 = l5_out.register_forward_hook(get_ftrs_l5)
    h8 = l8_out.register_forward_hook(get_ftrs_l8)

    fileNm1 = 'neural_activity_l' + str(2) + '.txt'
    fileNm2 = 'neural_activity_l' + str(5) + '.txt'
    fileNm3 = 'neural_activity_l' + str(8) + '.txt'
    np.savetxt(fileNm1, l2_out_copy, fmt='%1.5f')
    np.savetxt(fileNm2, l5_out_copy, fmt='%1.5f')
    np.savetxt(fileNm3, l8_out_copy, fmt='%1.5f')
    '''
