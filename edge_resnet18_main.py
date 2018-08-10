import shutil
import argparse
import os
import time
import cv2
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
import torchvision
import resnet_cifar
import edge_resnet_imagenet
#from resnet_cifar import ResNet as ResNet_cifar 
#from resnet_imagenet import ResNet as ResNet_imagenet
#from resnet_cifar import BasicBlock as BasicBlock_cifar 
#from resnet_imagenet import BasicBlock as BasicBlock_imagenet
#from resnet_cifar import Bottleneck  as Bottleneck_cifar 
#from resnet_imagenet import Bottleneck as Bottleneck_imagenet
#import torch.utils.model_zoo as model_zoo

parser = argparse.ArgumentParser(description ='PyTorch ResNet_imagenet')

parser.add_argument('--epochs', default = 150, type=int, metavar='N', help= 'epoch default =74')
parser.add_argument('--dataset', default= 'cifar10', type=str, help='dataset of cifar10 or imagenet, default is cifar10')
parser.add_argument('--start-epoch', default =0, type=int, metavar='N', help='default is 0')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='learning rate, default is 0.01')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum, default =0.9')
parser.add_argument('--weight-decay', '--wd', default = 1e-4, type=float, metavar='W', help='weight decay , default 1e-4')
parser.add_argument('-e', '--evaluate', action='store_true', help='evaulate model, default true')
parser.add_argument('--no-cuda', action='store_true', default = False, help='when not using cuda, default =false')
parser.add_argument('--batch_size', default = 256, type=int, help='default batchsize =50')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')



best_prec1=0

def main():

    path_current = os.path.dirname(os.path.realpath(__file__))
    path_subdir = 'edge_resnet_dataset'
    data_filename = 'edge_resnet18_dataset.txt'

    path_file = os.path.join(path_current,path_subdir,data_filename)
    f=open(path_file, 'w')

    global args, best_prec1
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device= torch.device("cuda" if use_cuda else "cpu")
    

    model = edge_resnet_imagenet.resnet18()

    state_dict18_untrained = model.state_dict()

    torch.save(state_dict18_untrained, 'edge_state_dict18_untrained.pt')

    model_weight_untrained = torch.load('edge_state_dict18_untrained.pt')

    



    model_trained = torchvision.models.resnet18(pretrained=True)


    state_dict18 = model_trained.state_dict()
    
    torch.save(state_dict18, 'edge_state_dict18.pt')

    a = torch.load('edge_state_dict18.pt')

#    a_init_weight = torch.randn(64,6,7,7)
    
#    a_init_weight = nn.Conv2d(6,64, kernel_size=7, stride =2, padding=3, bias=False)

#    nn.init.kaiming_normal_(a_init_weight.weight, mode='fan_out', nonlinearity='relu')

    a['conv1.weight'].data = model_weight_untrained['conv1.weight'].data

    model.load_state_dict(a)
    
    model_state_dict18 = model.state_dict()
    
#    for k, v in model_state_dict18.items():
#        print(k, v.size())

#    function my_transform(x):
#        canny = cv2.canny(x)
#        x = normalize(x)
#        return cat([x, canny])

#    transforms.lambda(my_transform)
    
    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
            model.parameters(),
            args.lr, 
            momentum = args.momentum, 
            weight_decay=args.weight_decay)
    
    train_losses = np.zeros((args.epochs))
    train_prec1s = np.zeros((args.epochs))
    eval_losses = np.zeros((args.epochs))
    eval_prec1s = np.zeros((args.epochs))
    x_epoch = np.zeros((args.epochs))


# optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    cudnn.benchmark = True

    traindir = os.path.join('../data/ILSVRC2012/', 'train')

    evaldir = os.path.join('../data/ILSVRC2012/', 'val')
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406,0.4], std=[0.229,0.224, 0.225,0.22])

    train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.Lambda(lambda img: makeEdge(img)),
                    transforms.ToTensor(),
                    normalize,
                    ]))
    
    train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True)

                

    eval_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                evaldir,
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.Lambda(lambda img: makeEdge(img)),
                    transforms.ToTensor(),
                    normalize,
                    ])
                ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True)


    for epoch in range(0, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        train_loss, train_prec1 = train(train_loader, model, criterion,optimizer, epoch, f)
        eval_loss, eval_prec1 = validate(eval_loader, model, criterion, f)


        train_losses[epoch] = train_loss
        train_prec1s[epoch] = train_prec1
        eval_losses[epoch] = eval_loss
        eval_prec1s[epoch] = eval_prec1
        x_epoch[epoch] = epoch


        # remember best prec@1 and save checkpoint
        is_best = eval_prec1 > best_prec1
        best_prec1 = max(eval_prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': data_filename,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
            'train loss': train_losses,
            'train Precision': train_prec1s,
            'test loss': eval_losses,
            'test Precision': eval_prec1s,
        }, is_best)



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
    plot_loss_filename = 'edge_resnet18loss.png'
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
    plot_prec_filename = 'edge_resnet18prec.png'
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
#        img = img.type(torch.cuda.FloatTensor).cuda()
        img = img.cuda()
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

def save_checkpoint(state, is_best, filename='checkpoint_edge_resnet18.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_edge_resnet18.pth.tar')


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

def makeEdge(img):

    img = cv2.GaussianBlur(np.asarray(img),(3,3),0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_8U)
    laplacian_cat = np.dstack((img,laplacian))
    return laplacian_cat

def adjust_learning_rate(optimizer, epoch):
    
    if epoch == 60:
        lr = args.lr /10
        for param_group in optimizer.param_groups:
            param_group['lr']=lr

    if epoch == 90:
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


