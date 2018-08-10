'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, stride):
        super(BasicBlock, self).__init__()
#        if stride ==2:
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        
#        else:
#            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential()
        if stride == 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,planes, kernel_size=1, stride=1,bias=False),
                nn.BatchNorm2d(planes)
            )
        if stride == 2:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
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


class ResNet(nn.Module):
    def __init__(self, block,n, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer1(block, 16, 1,n)
        self.layer2 = self._make_layer2(block, 32, 2,n)
        self.layer3 = self._make_layer2(block, 64, 2,n)
#        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#        self.pool1 = nn.Con2d(16,32, kernel_size=3, stride =2, padding=1, bias = False)
#        self.pool2 = nn.Con2d(32,64, kernel_size=3, stride =2, padding=1, bias = False)
#        self.pool3 = nn.Con2d(16,32, kernel_size=3, stride =2, padding=1, bias = False)
        self.relu = nn.ReLU()
        self.globalpool2d = nn.AvgPool2d(8,stride=0, padding=0) 
        self.linear = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        with open('model.txt', 'w') as f:
            print(self, file=f)

    def _make_layer1(self, block, planes, stride,n):
       # strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i in range(n):
            layers.append(block(self.in_planes, planes, 1))
#                if i ==n-1:
#                    layers.append(block(self.in_planes,planes, 2))
        self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_layer2(self, block, planes, stride,n):
       # strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i in range(n):
            if i ==0:
                layers.append(block(self.in_planes/2,planes, 2))
            else:
                layers.append(block(self.in_planes, planes, 1))
        self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
#        print(out.size())
        out = self.globalpool2d(out)
#        print(out.size())
        out = out.view(out.size(0), -1)
#        print(out.size())
        out = self.linear(out)
        return out


def ResNet_cifar(n):
    return ResNet(BasicBlock,n)

