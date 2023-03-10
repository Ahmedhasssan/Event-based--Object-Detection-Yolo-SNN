import torch
import torch.nn as nn
from torch import Tensor
import math
from models.layers import *
from models.operations import *

thre=0.3
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
