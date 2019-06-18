import time
import torch
import torch.nn as nn

def adjust_learning_rate(optimizer, epoch, LR):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = LR * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    "Computes and stores the average and current value"
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1,-1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(train_loader, model, criterion, optimizer, epoch, device):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    total_iters = len(train_loader)

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        bs, ncrops, c, h, w = input.size()

        # transfer input to appropriate device
        input, target = input.to(device), target.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # compute output
        #output = model(input)
        output = model(input.view(-1,c,h,w)) # fuse batch size and ncrops
        output_avg = output.view(bs, ncrops, -1).mean(1) #avg over crops
        output = output_avg
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # backprop and SGD
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 250 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                  epoch, i, len(train_loader), batch_time=batch_time,
                  loss=losses, top1=top1))

def validate(val_loader, model, criterion, epoch, device, compute_thresholds=False, hooks=None):
    """ Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    #print(model)

    end = time.time()
    max_vals = None
    max_in = torch.zeros(1).to(device)
    if compute_thresholds:
        max_vals = torch.zeros(len(hooks)+1).to(device)
    for i, (input, target) in enumerate(val_loader):
        input, target = input.to(device), target.to(device)
        max_in = torch.max(max_in, torch.max(input))

        '''
        import numpy
        for param in model.parameters():
            print (param.data.size())
            wts = param.data.cpu().numpy()
            if len(wts.shape) > 1:
                print('wt max {}\twt min{}'.format(numpy.amax(wts), numpy.amin(wts)))
        '''

        # compute output
        with torch.no_grad():
            output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 25 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1, top5=top5))

        index = None
        if compute_thresholds:
            #for j in range(len(hooks)):
            for j, hook in enumerate(hooks):
                '''
                if len(hook.output.size()) > 2:
                    index = (0,0,3,0)
                else:
                    index = (0,0)
                #print(type(hook.input[0]))
                layer_out = hook.output[index]
                if len(hook.input[0].size()) > 2:
                    index = (0,0,3,0)
                else:
                    index = (0,0)
                layer_in = hook.input[0][index]
                if j > 10:
                    break
                print(j, layer_in, layer_out, hook.input[0].size(), hook.output.size())
                '''
                max_vals[j+1] = torch.max(max_vals[j+1], torch.max(hook.output))
            max_vals[0] = max_in

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    print(' * Prec@5 {top5.avg:.3f}'.format(top5=top5))

    return (top1.avg, max_vals)


