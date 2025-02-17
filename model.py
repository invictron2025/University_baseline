import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torchvision.models import ResNet50_Weights
######################################################################

def weights_init_kaiming(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    if isinstance(m, nn.Linear):
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        
        x = self.classifier(x)
        return x





# Define the ResNet50-based Model
class ft_net(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        self.model = model_ft if init_model is None else init_model.model
        self.model.avgpool2 = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
       
        x = self.model.avgpool2(x)
      

         
        #x = self.classifier(x)
        return x.squeeze(-1).squeeze(-1)




class three_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride=2, pool='avg', share_weight=False, circle=False):
        super(three_view_net, self).__init__()
        self.model_1 = ft_net(class_num, stride=stride, pool=pool)
        self.model_2 = ft_net(class_num, stride=stride, pool=pool)
        self.model_3=self.model_1
        self.classifier = ClassBlock(2048, class_num, droprate, return_f=circle)

      

    def forward(self, x1, x2, x3, x4=None):
        process = lambda x, model: self.classifier(model(x)) if x is not None else None

        y1 = process(x1, self.model_1)
        y2 = process(x2, self.model_2)
        y3 = process(x3, self.model_1)  # model_3 is the same as model_1
        y4 = process(x4, self.model_2) if x4 is not None else None

        return (y1, y2, y3, y4) if x4 is not None else (y1, y2, y3)

