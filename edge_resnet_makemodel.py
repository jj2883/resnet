import torchvision
import torch


resnet18 = torchvision.models.resnet18(pretrained=True)

resnet34 = torchvision.models.resnet34(pretrained=True)
resnet50 = torchvision.models.resnet50(pretrained=True)
resnet101 = torchvision.models.resnet101(pretrained=True)


state_dict18 = resnet18.state_dict()
#for k, v in state_dict18.items():
#    print(k, v.size())


torch.save(state_dict18, 'edge_state_dict18.pt')

a = torch.load('edge_state_dict18.pt')

a['conv1.weight'] = torch.randn(64,6,7,7)

#torch.save(a, 'edge_state_dict18.pt')

for k, v in a.items():
    print(k, v.size())

