import argparse
import os
import time
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.parallel
import torch.nn.functional as F

import resnet_cifar
from resnet_cifar import ResNet as ResNet_cifar 
from resnet_imagenet import ResNet as ResNet_imagenet
from resnet_cifar import BasicBlock as BasicBlock_cifar 
from resnet_imagenet import BasicBlock as BasicBlock_imagenet
from resnet_cifar import Bottleneck  as Bottleneck_cifar 
from resnet_imagenet import Bottleneck as Bottleneck_imagenet
#import torch.utils.model_zoo as model_zoo

parser = argparse.ArgumentParser(description ='PyTorch ResNet_imagenet')

parser.add_argument('--epochs', default = 74, type=int, metavar='N', help= 'epoch default =74')
parser.add_argument('--dataset', default= 'cifar10', type=str, help='dataset of cifar10 or imagenet, default is cifar10')
parser.add_argument('--start-epoch', default =0, type=int, metavar='N', help='default is 0')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='learning rate, default is 0.01')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum, default =0.9')
parser.add_argument('--weight-decay', '--wd', default = 1e-4, type=float, metavar='W', help='weight decay , default 1e-4')
parser.add_argument('-e', '--evaluate', action='store_true', help='evaulate model, default true')
parser.add_argument('--no-cuda', action='store_true', default = False, help='when not using cuda, default =false')
parser.add_argument('--batch_size', default = 50, type=int, help='default batchsize =50')





def main():

    path_current = os.path.dirname(os.path.realpath(__file__))
    path_subdir = 'resnet_dataset'
    data_filename = 'Resnet_dataset.txt'

    path_file = os.path.join(path_current,path_subdir,data_filename)
    f=open(path_file, 'w')

    global args, best_prec1
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device= torch.device("cuda" if use_cuda else "cpu")
    
    model = ResNet_cifar(3,**kwargs)


    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
            model.parameters(),
            args.lr, 
            momentum = args.momentum, 
            weight_decay=args.weight_decay)

    cudnn.benchmark = True
    
    train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                '../data',
                train=True,
                download=True,
                transform = transforms.Compose([
                    transforms.RandomCrop(32,4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485,0.456, 0.406],
                        std=[0.229,0.224,0.225]),
                    ])
                ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True)
    

    eval_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                '../data',
                train=False,
                download=True,
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485,0.456, 0.406],
                        std=[0.229,0.224,0.225]),
                    ])
                ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True)

    train_losses = np.zeros((args.epochs))
    train_prec1s = np.zeros((args.epochs))
    eval_losses = np.zeros((args.epochs))
    eval_prec1s = np.zeros((args.epochs))
    x_epoch = np.zeros((args.epochs))

    for epoch in range(0, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        train_loss, train_prec1 = train(train_loader, model, criterion,optimizer, epoch, f)
        eval_loss, eval_prec1 = validate(eval_loader, model, criterion, f)

        train_losses[epoch] = train_loss
        train_prec1s[epoch] = train_prec1
        eval_losses[epoch] = eval_loss
        eval_prec1s[epoch] = eval_prec1
        x_epoch[epoch] = epoch

    plt.clf()
    plt.close()
    fig_loss = plt.figure()
    ax_loss = fig_loss.add_subplot(1,1,1)
    ax_loss.plot(x_epoch, train_losses, label='Train Loss')
    ax_loss.plot(x_epoch, eval_losses, label='Test Loss')
    ax_loss.legend(loc=1)
    ax_loss.set_xlabel('epoch')
    ax_loss.set_ylabel('loss')
    ax_loss.set_title('Loss of Train and Test')
    plot_loss_filename = 'resnetloss.png'
    path_loss_file = os.path.join(path_current, path_subdir, plot_loss_filename)
    fig_loss.savefig(path_loss_file)


    plt.clf()
    plt.close()
    fig_prec = plt.figure()
    ax_prec = fig_prec.add_subplot(1,1,1)
    ax_prec.plot(x_epoch, train_prec1s, label='Train Best1')
    ax_prec.plot(x_epoch, eval_prec1s, label='Test Best1')
    ax_prec.legend(loc=1)
    ax_prec.set_xlabel('epoch')
    ax_prec.set_ylabel('Best1 Precision')
    ax_prec.set_title('Best1 Precision of Train and Test')
    plot_prec_filename = 'resnetprec.png'
    path_prec_file = os.path.join(path_current, path_subdir, plot_prec_filename)
    fig_prec.savefig(path_prec_file)

    f.close()


def train(train_loader, model, criterion, optimizer, epoch, f):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    
    for i, (img, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        target = target.cuda()
        img = img.type(torch.cuda.FloatTensor).cuda()
        output = model(img)
        loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1,5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1[0], img.size(0))
        top5.update(prec1[0], img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 ==0:
            f.write('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f}({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.2f}({top1.avg:.2f})\t'
                    'Prec@5 {top5.val:.2f}({top5.avg:.2f})\r\n'.format(
                        epoch, i,len(train_loader), 
                        loss=losses, top1=top1, top5=top5))

            print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f}({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.2f}({top1.avg:.2f})\t'
                    'Prec@5 {top5.val:.2f}({top5.avg:.2f})'.format(
                        epoch, i,len(train_loader), 
                        loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg

def validate(eval_loader, model, criterion, f):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        
        for i, (img, target) in enumerate(eval_loader):
    
            target = target.cuda()
            img = img.cuda()
    
            output = model(img)
            loss = criterion(output, target)
    
            prec1, prec5 = accuracy(output, target, topk=(1,5))
            losses.update(loss.item(), img.size(0))
            top1.update(prec1[0], img.size(0))
            top5.update(prec1[0], img.size(0))
    
    
            batch_time.update(time.time() - end)
            end = time.time()
    
            if i % 10 ==0:
                f.write('Test: [{0}/{1}\t'
                        'Loss {loss.val:.4f}({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.2f}({top1.avg:.2f})\t'
                        'Prec@5 {top5.val:.2f}({top5.avg:.2f})\r\n'.format(
                            i,len(eval_loader), 
                            loss=losses, top1=top1, top5=top5))
    
                print('Test: [{0}/{1}]\t'
                        'Loss {loss.val:.4f}({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.2f}({top1.avg:.2f})\t'
                        'Prec@5 {top5.val:.2f}({top5.avg:.2f})'.format(
                            i,len(eval_loader), 
                            loss=losses, top1=top1, top5=top5))
    
                f.write('***Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\r\n'
                        .format(top1=top1, top5=top5))
                print('***Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                        .format(top1=top1, top5=top5))
    return losses.avg, top1.avg


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count

def adjust_learning_rate(optimizer, epoch):
    
    if epoch == 30:
        lr = args.lr /10
        for param_group in optimizer.param_groups:
            param_group['lr']=lr

    if epoch == 54:
        lr = args.lr /100
        for param_group in optimizer.param_groups:
            param_group['lr']=lr

def accuracy(output, target, topk=(1,)):

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1,-1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
if __name__ == '__main__':
    main()


