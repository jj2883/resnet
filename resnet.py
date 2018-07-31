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
#import torch.utils.model_zoo as model_zoo

parser = argparse.ArgumentParser(description ='PyTorch ResNet_imagenet')

parser.add_argument('--epochs', default = 64000, type=int, metavar='N', help= 'epoch default =74')
parser.add_argument('--dataset', default= 'cifar10', type=str, help='dataset of cifar10 or imagenet, default is cifar10')
parser.add_argument('--start-epoch', default =0, type=int, metavar='N', help='default is 0')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='learning rate, default is 0.01')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum, default =0.9')
parser.add_argument('--weight-decay', '--wd', default = 1e-4, type=float, metavar='W', help='weight decay , default 1e-4')
parser.add_argument('-e', '--evaluate', action='store_true', help='evaulate model, default true')
parser.add_argument('--no-cuda', action='store_true', default = False, help='when not using cuda, default =false')
parser.add_argument('--resnet', default = 'resnet50', type=str, help='choose from resnet18, resnet34, 50, 101, 152, default is 50')
parser.add_argument('--batch_size', default = 50, type=int, help='default batchsize =50')

#__all__ = ['ResNet_imagenet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#           'resnet152']


#model_urls = {
#    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
#}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock_imagenet(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_imagenet, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck_imagenet(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_imagenet, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_imagenet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet_imagenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x









class BasicBlock_cifar(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_cifar, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck_cifar(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_cifar, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_cifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_cifar, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            type(m)
#
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)        
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out




def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet_imagenet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if args.dataset == 'cifar10':
        model = ResNet_cifar(BasicBlock_cifar,[2, 2, 2, 2], **kwargs)
    else:
        model = ResNet_imagenet(BasicBlock_imagenet, [2, 2, 2, 2], **kwargs)
#    if pretrained:
#        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet_imagenet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if args.dataset == 'cifar10':
        model = ResNet_cifar(BasicBlock_cifar, [3, 4, 6, 3], **kwargs)
    else:
        model = ResNet_imagenet(BasicBlock_imagenet, [3, 4, 6, 3], **kwargs)
#    if pretrained:
#        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet_imagenet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if args.dataset == 'cifar10':
        model = ResNet_cifar(Bottleneck_cifar, [3, 4, 6, 3], **kwargs)
    else:
        model = ResNet_imagenet(Bottleneck_imagenet, [3, 4, 6, 3], **kwargs)
#    if pretrained:
#        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet_imagenet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if args.dataset == 'cifar10':
        model = ResNet_cifar(Bottleneck_cifar, [3, 4, 23, 3], **kwargs)
    else:
        model = ResNet_imagenet(Bottleneck_imagenet, [3, 4, 23, 3], **kwargs)
#    if pretrained:
#        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet_imagenet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_cifar(Bottleneck_cifar, [3, 8, 36, 3], **kwargs)
    model = ResNet_imagenet(Bottleneck_imagenet, [3, 8, 36, 3], **kwargs)
#    if pretrained:
#        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model



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

    if args.resnet == 'resnet18':
        model = resnet18()
    elif args.resnet == 'resnet34':
        model = resnet34()
    elif args.resnet == 'resnet50':
        model = resnet50()
    elif args.resnet == 'resnet101':
        model = resnet101()
    elif args.resnet == 'resnet152':
        model = resnet152()
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
    
    if epoch == 34000:
        lr = args.lr /10
        for param_group in optimizer.param_groups:
            param_group['lr']=lr

    if epoch == 48000:
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


