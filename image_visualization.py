import os
import abc
import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import visualizations as visualizations
import logging
from PIL import Image
from modules import *
from model import QUAutoencoder
import model
import dataset
from loader import Loader
from models.dense_object_det import DenseObjectDet, Yolov2, Yolov2_SNN
from models.yolo_loss import yoloLoss
from models.yolo_detection import yoloDetect
from models.yolo_detection import nonMaxSuppression
from statistics_pascalvoc import BoundingBoxes, BoundingBox, BBType, VOC_Evaluator, MethodAveragePrecision

class data_build:
    def __init__(self, root, classes, height, width, event, data):

        self.train_loader = None
        self.test_loader = None
        self.num_cpu_workers=4
        self.gpu_device=0,1,3
        CUDA_VISIBLE_DEVICES=0,1,3

        self.model_input_size = torch.tensor([224, 288])
        self.object_classes = classes
        self.model=model.QUAutoencoder()
        ####self.object_det = DenseObjectDet()
        #self.object_det =Yolov2()
        self.object_det =Yolov2_SNN()
        self.model.cuda()
        self.dataset_path=root
        

        self.steps_lr=[500, 1000, 1500]
        self.factor_lr=0.1

        self.height=height
        self.width=width

        self.nr_events_window=25000
        self.event_representation=event
        self.dataset_name= data

        self.batch_size=10

        self.dataset_builder = dataset.getDataloader(self.dataset_name)
        self.dataset_loader = Loader
        """
        Creates the validation and the training data based on the lists specified in the config/settings.yaml file.
        """
        train_dataset = self.dataset_builder(self.dataset_path,
                                                self.object_classes,
                                                self.height,
                                                self.width,
                                                self.nr_events_window,
                                                #augmentation=True,
                                                mode='training',
                                                event_representation=self.event_representation)

        self.nr_classes = train_dataset.nr_classes
        self.object_classes = train_dataset.object_classes

        test_dataset = self.dataset_builder(self.dataset_path,
                                            self.object_classes,
                                            self.height,
                                            self.width,
                                            self.nr_events_window,
                                            mode='test',
                                            event_representation=self.event_representation)
        self.train_loader = self.dataset_loader(train_dataset, batch_size=self.batch_size,
                                                num_workers=self.num_cpu_workers, pin_memory=False)
        self.test_loader = self.dataset_loader(test_dataset, batch_size=self.batch_size,
                                                num_workers=self.num_cpu_workers, pin_memory=False)

class View_Dataloader(data_build):
    def data(self):
        #self.model_input_size = torch.tensor([223, 287])
        self.model_input_size = torch.tensor([240, 304])
        for i_batch, sample_batched in enumerate(self.train_loader):
            event, bounding_box, Histogram = sample_batched
            for h in range(3):
                histogram = Histogram[:,h,:,:,:]
                #histogram = Histogram
                histogram = torch.nn.functional.interpolate(histogram.permute(0, 3, 1, 2),
                                                                   torch.Size(self.model_input_size))
                #histogram = histogram.permute(0, 3, 1, 2)
                image = histogram[0, :, :, :].permute(1, 2, 0).cpu().int().numpy()
                # image = visualizations.visualizeHistogram(histogram[0, :, :, :].permute(1, 2, 0).cpu().int().numpy())
                # print(np.unique(image))
                # bounding_box[:, :, [0, 2]] = (bounding_box[:, :, [0, 2]] * self.model_input_size[1].float()
                #                             / 304).long()
                # bounding_box[:, :, [1, 3]] = (bounding_box[:, :, [1, 3]] * self.model_input_size[0].float()
                #                             / 240).long()
                
                bounding_box[:, :, [0, 2]] = (bounding_box[:, :, [0, 2]]).long()
                bounding_box[:, :, [1, 3]] = (bounding_box[:, :, [1, 3]]).long()

                image = visualizations.visualizeHistogram(histogram[0, :, :, :].permute(1, 2, 0).cpu().int().numpy())
                image = visualizations.drawBoundingBoxes(image, bounding_box[0, :, :].cpu().numpy(),
                                                            class_name=[self.object_classes[i]
                                                                        for i in bounding_box[0, :, -1]],
                                                            ground_truth=True, rescale_image=True)
                
                plt.imshow(image)
                plt.savefig("images.png")
                print(i_batch)

if __name__=='__main__':
    root="/home2/ahasssan/Prophesee-Gen4/Gen1/"
    classes= 'all'
    height=240
    width=304
    data='Prophesee'
    event='frames'  # ['histogram', 'event_queue', 'frames']
    view=View_Dataloader(root=root, classes=classes, height=height, width=width, data=data, event=event)
    view.data()