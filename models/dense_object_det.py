import torch
import torch.nn as nn
from torch import Tensor
import math
from models.layers import *
from models.operations import *

thre=0.3

config = [
    [2, 32, 1, 1],
    'M',
    [32, 64, 1, 1],
    'M',
    [64, 128, 1, 1],
    [128, 64, 1, 1],
    [64, 128, 1, 1],
    'M',
    [128, 256, 1, 1],
    [256, 128, 1, 1],
    [128, 256, 1, 1],
    'M',
    [256, 512, 1, 1],
    [512, 256, 1, 1],
    [256, 512, 1, 1],
    [512, 256, 1, 1],
    [256, 512, 1, 1],
    'M',
    [512, 1024, 1, 1],
    [1024, 512, 1, 1],
    [512, 1024, 1, 1],
    [1024, 512, 1, 1],
    [512, 1024, 1, 1],
    'M',
    [1024, 1024, 1, 1],
    'M'
]

class DenseObjectDet(nn.Module):
    def __init__(self, nr_classes=2, in_c=2, nr_box=2, small_out_map=True):
        super().__init__()
        self.nr_box = nr_box
        self.nr_classes = nr_classes

        self.kernel_size = 3
        self.conv_layers = nn.Sequential(
            self.conv_block(in_c=in_c, out_c=16),
            self.conv_block(in_c=16, out_c=32),
            self.conv_block(in_c=32, out_c=64),
            self.conv_block(in_c=64, out_c=128),
            self.conv_block(in_c=128, out_c=256, max_pool=False),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=self.kernel_size, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        #if small_out_map:
        #    self.cnn_spatial_output_size = [5, 7]
        #else:
        self.cnn_spatial_output_size = [6, 8]

        spatial_size_product = self.cnn_spatial_output_size[0] * self.cnn_spatial_output_size[1]

        self.linear_input_features = spatial_size_product * 512
        self.linear_1 = nn.Linear(self.linear_input_features, 1024)
        self.linear_2 = nn.Linear(1024, spatial_size_product*(nr_classes + 5*self.nr_box))

    def conv_block(self, in_c, out_c, max_pool=True):
        if max_pool:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=self.kernel_size, padding=(1, 1), bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, kernel_size=self.kernel_size, padding=(1, 1), bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=self.kernel_size, stride=2)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=self.kernel_size, padding=(1, 1), bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, kernel_size=self.kernel_size, padding=(1, 1), bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(-1, self.linear_input_features)
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.linear_2(x)
        x = x.view([-1] + self.cnn_spatial_output_size + [(self.nr_classes + 5*self.nr_box)])
        return x


class Yolov2(nn.Module):
    def __init__(self, nr_classes=2, in_c=2, nr_box=2, small_out_map=True):
        super().__init__()
        self.nr_box = nr_box
        self.nr_classes = nr_classes

        self.kernel_size1 = 3
        self.kernel_size2 = 1
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_c, 32, kernel_size=self.kernel_size1, padding=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=self.kernel_size1, padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            self.conv_block1(in_c=64, out_c=128),
            self.conv_block1(in_c=128, out_c=256),
            self.conv_block2(in_c=256, out_c=512),
            self.conv_block2(in_c=512, out_c=1024),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=self.kernel_size2, stride=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.AvgPool2d(2)
        )

        #if small_out_map:
        #    self.cnn_spatial_output_size = [5, 7]
        #else:
        self.cnn_spatial_output_size = [6, 8]

        spatial_size_product = self.cnn_spatial_output_size[0] * self.cnn_spatial_output_size[1]

        self.linear_input_features = spatial_size_product * 192
        self.linear_1 = nn.Linear(self.linear_input_features, 512)
        self.linear_2 = nn.Linear(512, spatial_size_product*(nr_classes + 5*self.nr_box))

    def conv_block1(self, in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=self.kernel_size1, padding=(1, 1), bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.Conv2d(out_c, in_c, kernel_size=self.kernel_size2, padding=(1, 1), bias=False),
                nn.BatchNorm2d(in_c),
                nn.ReLU(),
                nn.Conv2d(in_c, out_c, kernel_size=self.kernel_size1, padding=(1, 1), bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

    def conv_block2(self, in_c, out_c, max_pool=True):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=self.kernel_size1, padding=(1, 1), bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.Conv2d(out_c, in_c, kernel_size=self.kernel_size2, padding=(1, 1), bias=False),
                nn.BatchNorm2d(in_c),
                nn.ReLU(),
                nn.Conv2d(in_c, out_c, kernel_size=self.kernel_size1, padding=(1, 1), bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.Conv2d(out_c, in_c, kernel_size=self.kernel_size2, padding=(1, 1), bias=False),
                nn.BatchNorm2d(in_c),
                nn.ReLU(),
                nn.Conv2d(in_c, out_c, kernel_size=self.kernel_size1, padding=(1, 1), bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )


    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(-1, self.linear_input_features)
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.linear_2(x)
        x = x.view([-1] + self.cnn_spatial_output_size + [(self.nr_classes + 5*self.nr_box)])
        return x


###### Yolov2 SNN Implemenation ########
class Yolov2_SNN(nn.Module):
    def __init__(self, nr_classes=2, in_c=2, nr_box=2, small_out_map=True):
        super().__init__()
        self.nr_box = nr_box
        self.nr_classes = nr_classes
        self.maxpool = SeqToANNContainer(nn.MaxPool2d(2))
        self.avgpool = SeqToANNContainer(nn.AvgPool2d(2))
        self.kernel_size1 = 3
        self.kernel_size2 = 1
        self.conv_layers = nn.Sequential(
            Layer(in_c,32,self.kernel_size1,1,1),
            self.maxpool,
            Layer(32,64,self.kernel_size1,1,1),
            self.maxpool,
            self.conv_block1(in_c=64, out_c=128),
            self.conv_block1(in_c=128, out_c=256),
            self.conv_block2(in_c=256, out_c=512),
            self.conv_block2(in_c=512, out_c=1024),
            Layer(1024,1024,self.kernel_size2,1,0),
            self.avgpool,
        )

        #if small_out_map:
        #    self.cnn_spatial_output_size = [5, 7]
        #else:
        self.cnn_spatial_output_size = [6, 8]

        spatial_size_product = self.cnn_spatial_output_size[0] * self.cnn_spatial_output_size[1]

        self.linear_input_features = spatial_size_product * 192   ##### For Image size of 128
        self.linear_1 = SeqToANNContainer(nn.Linear(self.linear_input_features, 512))  ##### For Image size of 128
        self.linear_2 = SeqToANNContainer(nn.Linear(512, spatial_size_product*(nr_classes + 5*self.nr_box)))  ##### For Image size of 128
        self.act = ZIFArchTan()


        # self.linear_input_features = spatial_size_product * 10240  ##### For Image size of [223, 287]
        # self.linear_1 = nn.Linear(self.linear_input_features, 512)
        # self.linear_2 = nn.Linear(512, spatial_size_product*(nr_classes + 5*self.nr_box))

    def conv_block1(self, in_c, out_c):
            return nn.Sequential(
                Layer(in_c,out_c,self.kernel_size1,1,1),
                Layer(out_c,in_c,self.kernel_size2,1,1),
                Layer(in_c,out_c,self.kernel_size1,1,1),
                self.maxpool,
            )

    def conv_block2(self, in_c, out_c, max_pool=True):
            return nn.Sequential(
                Layer(in_c, out_c, self.kernel_size1,1,1),
                Layer(out_c, in_c, self.kernel_size2,1,1),
                Layer(in_c,out_c,self.kernel_size1,1,1),
                Layer(out_c, in_c, self.kernel_size2,1,1),
                Layer(in_c,out_c,self.kernel_size1,1,1),
                self.maxpool,
            )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(-1, self.linear_input_features)
        x = x.unsqueeze(0)
        x = self.linear_1(x)
        #x = self.act(x)
        x = torch.relu(x)
        x = self.linear_2(x)
        x = x.squeeze(0)
        x = x.view([-1] + self.cnn_spatial_output_size + [(self.nr_classes + 5*self.nr_box)])
        return x

######## YoloV5 model #####

class Yolov5(nn.Module):
    def __init__(self, nr_classes=2, in_c=2, nr_box=2, small_out_map=True):
        super().__init__()
        self.nr_box = nr_box
        self.nr_classes = nr_classes

        self.kernel_size1 = 3
        self.kernel_size2 = 1
        self.backbone1 = nn.Sequential(
            nn.Conv2d(in_c, 64, kernel_size=self.kernel_size1, stride =1, padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            #nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=self.kernel_size1, stride =2, padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            #nn.MaxPool2d(2),
        )
        self.backbone2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=self.kernel_size1, stride =2, padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #nn.MaxPool2d(2),
        )
        self.backbone3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=self.kernel_size1, stride =2, padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            #nn.MaxPool2d(2),
        )
        self.backbone4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=self.kernel_size1, stride =2, padding=(1, 1), bias=False),
            nn.BatchNorm2d(1024),
            nn.SiLU(),
            #nn.MaxPool2d(2),
        )
        self.head1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=self.kernel_size2, stride =1, padding=(0, 0), bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(),
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.head2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=self.kernel_size2, stride = 1, padding=(0, 0), bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(),
        )
        self.head3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=self.kernel_size1, stride =2, padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(),
        )
        self.head4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=self.kernel_size1, stride =2, padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(),
        )

        #if small_out_map:
        #    self.cnn_spatial_output_size = [5, 7]
        #else:
        self.cnn_spatial_output_size = [6, 8]

        spatial_size_product = self.cnn_spatial_output_size[0] * self.cnn_spatial_output_size[1]

        self.linear_input_features = spatial_size_product * 192
        self.bottleneck_1 = BottleneckCSP(128, 128, 1)
        self.bottleneck_2 = BottleneckCSP(256, 256, 1)
        self.bottleneck_3 = BottleneckCSP(512, 512, 1)
        self.bottleneck_4 = BottleneckCSP(1024, 1024, 1, False)
        self.bottleneck_5 = BottleneckCSP(1024, 512, 1, False)
        self.bottleneck_6 = BottleneckCSP(512, 256, 1, False)
        self.bottleneck_7 = BottleneckCSP(512, 256, 1, False)
        self.bottleneck_8 = BottleneckCSP(1024, 1024, 1, False)
        self.SPP = SPP(1024, 1024)
        #self.detect = Detect(2, spatial_size_product*(nr_classes + 5*self.nr_box))
        self.linear_2 = nn.Linear(512, spatial_size_product*(nr_classes + 5*self.nr_box))

    def forward(self, x):
        x = self.backbone1(x)
        csp1 = self.bottleneck_1(x)
        x = self.backbone2(csp1)
        csp2 = self.bottleneck_2(x)
        x = self.backbone3(csp2)
        csp3 = self.bottleneck_3(x)
        x = self.backbone4(csp3)
        x = self.SPP(x)
        x = self.bottleneck_4(x)
        c1 = self.head1(x)
        x = self.up(c1)
        x = torch.cat((x,csp3),1)
        x = self.bottleneck_5(x)
        c2 = self.head2(x)
        x = self.up(c2)
        x = torch.cat((x,csp2),1)
        x = self.bottleneck_6(x)
        x1 = x.reshape([-1] + self.cnn_spatial_output_size + [(self.nr_classes + 5*self.nr_box)])
        x = self.head3(x)
        x = torch.cat((x,c2),1)
        x = self.bottleneck_7(x)
        x2 = x.reshape([-1] + self.cnn_spatial_output_size + [(self.nr_classes + 5*self.nr_box)])
        x = self.head4(x)
        x = torch.cat((x,c1),1)
        x = self.bottleneck_8(x)
        x3 = x.reshape([-1] + self.cnn_spatial_output_size + [(self.nr_classes + 5*self.nr_box)])
        #x = self.detect()
        return x3

######## YoloV5 model #####

class Yolov3(nn.Module):
    def __init__(self, nr_classes=2, in_c=2, nr_box=2, small_out_map=True):
        super().__init__()
        self.nr_box = nr_box
        self.nr_classes = nr_classes

        self.kernel_size1 = 3
        self.kernel_size2 = 1
        self.backbone1 = nn.Sequential(
            nn.Conv2d(in_c, 64, kernel_size=self.kernel_size1, stride =1, padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            #nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=self.kernel_size1, stride =2, padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            #nn.MaxPool2d(2),
        )
        self.backbone2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=self.kernel_size1, stride =2, padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #nn.MaxPool2d(2),
        )
        self.backbone3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=self.kernel_size1, stride =2, padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            #nn.MaxPool2d(2),
        )
        self.backbone4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=self.kernel_size1, stride =2, padding=(1, 1), bias=False),
            nn.BatchNorm2d(1024),
            nn.SiLU(),
            #nn.MaxPool2d(2),
        )
        self.head1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=self.kernel_size2, stride =1, padding=(0, 0), bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(),
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.head2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=self.kernel_size2, stride = 1, padding=(0, 0), bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(),
        )
        self.head3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=self.kernel_size1, stride =2, padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(),
        )
        self.head4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=self.kernel_size1, stride =2, padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(),
        )

        #if small_out_map:
        #    self.cnn_spatial_output_size = [5, 7]
        #else:
        self.cnn_spatial_output_size = [6, 8]

        spatial_size_product = self.cnn_spatial_output_size[0] * self.cnn_spatial_output_size[1]

        self.linear_input_features = spatial_size_product * 192
        self.bottleneck_1 = nn.Linear(128,128)
        self.bottleneck_2 = nn.Linear(256,128)
        self.bottleneck_3 = nn.Linear(512,512)
        self.bottleneck_4 = nn.Linear(1024,1024)
        self.bottleneck_5 = nn.Linear(1024,512)
        self.bottleneck_6 = nn.Linear(512,256)
        self.bottleneck_7 = BottleneckCSP(512, 256, 3, False)
        self.bottleneck_8 = BottleneckCSP(1024, 1024, 3, False)
        self.SPP = SPP(1024, 1024)
        #self.detect = Detect(2, spatial_size_product*(nr_classes + 5*self.nr_box))
        self.linear_2 = nn.Linear(512, spatial_size_product*(nr_classes + 5*self.nr_box))

    def forward(self, x):
        x = self.backbone1(x)
        csp1 = self.bottleneck_1(x)
        x = self.backbone2(csp1)
        csp2 = self.bottleneck_2(x)
        x = self.backbone3(csp2)
        csp3 = self.bottleneck_3(x)
        x = self.backbone4(csp3)
        x = self.SPP(x)
        x = self.bottleneck_4(x)
        c1 = self.head1(x)
        x = self.up(c1)
        x = torch.cat((x,csp3),1)
        x = self.bottleneck_5(x)
        c2 = self.head2(x)
        x = self.up(c2)
        x = torch.cat((x,csp2),1)
        x = self.bottleneck_6(x)
        x1 = x.reshape([-1] + self.cnn_spatial_output_size + [(self.nr_classes + 5*self.nr_box)])
        x = self.head3(x)
        x = torch.cat((x,c2),1)
        x = self.bottleneck_7(x)
        x2 = x.reshape([-1] + self.cnn_spatial_output_size + [(self.nr_classes + 5*self.nr_box)])
        x = self.head4(x)
        x = torch.cat((x,c1),1)
        x = self.bottleneck_8(x)
        x3 = x.reshape([-1] + self.cnn_spatial_output_size + [(self.nr_classes + 5*self.nr_box)])
        #x = self.detect()
        return x1

