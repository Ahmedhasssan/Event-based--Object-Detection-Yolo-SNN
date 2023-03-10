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

from modules import *
from model import QUAutoencoder
import model
import dataset
import argparse
import shutil
import os
import time
import logging
from loader import Loader
from models.dense_object_det import DenseObjectDet, Yolov2, Yolov2_SNN, Yolov5
from models.yolo_loss import yoloLoss
from models.yolo_detection import yoloDetect
from models.yolo_detection import nonMaxSuppression
from statistics_pascalvoc import BoundingBoxes, BoundingBox, BBType, VOC_Evaluator, MethodAveragePrecision
#import utils.visualizations as visualizations
from models.layers import ZIFArchTan

parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training')
parser.add_argument('-j',
                    '--workers',
                    default=10,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 10)')
parser.add_argument('--frequency',
                    default=500,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 500)')
parser.add_argument('--event',
                    default='histogram',
                    type=str,
                    metavar='N',
                    help='print frequency (default: 500)')
parser.add_argument('--epochs',
                    default=200,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch-size',
                    default=128,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.001,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('-p',
                    '--print-freq',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed',
                    default=1000,
                    type=int,
                    help='seed for initializing training. ')
parser.add_argument('--T',
                    default=20,
                    type=int,
                    metavar='N',
                    help='snn simulation time (default: 20)')
parser.add_argument('--TET',
                    default=True,
                    type=bool,
                    metavar='N',
                    help='if use Temporal Efficient Training (default: True)')
parser.add_argument('--lvth',
                    default=False,
                    type=bool,
                    metavar='N',
                    help='if use learnable threshold (default: True)')
parser.add_argument('--lamb',
                    default=0.90,
                    type=float,
                    metavar='N',
                    help='adjust the norm factor to avoid outlier (default: 0.0)')
parser.add_argument('--dataset',
                    default='Gen1',
                    type=str,
                    metavar='N',
                    help='dataset')
parser.add_argument('--model',
                    default='Yolov2',
                    type=str,
                    metavar='N',
                    help='model for training')
parser.add_argument('--architecture',
                    default='object-detection',
                    type=str,
                    metavar='N',
                    help='Choice of object detection process')
parser.add_argument('--process',
                    default='train',
                    type=str,
                    metavar='N',
                    help='Choice of process, either train or test')
parser.add_argument('--save_path', 
                    type=str, 
                    default='./train/', 
                    help='Folder to save checkpoints and log.')
parser.add_argument('--save_images', 
                    type=str, 
                    default='./train/', 
                    help='Folder to save iamges.')
args = parser.parse_args()

logger = logging.getLogger('training')
def glasso_global_mp(var, dim=0, thre=0.0):
    if len(var.size()) == 4:
        var = var.contiguous().view((var.size(0), var.size(1) * var.size(2) * var.size(3)))

    a = var.pow(2).sum(dim=dim).pow(1/2)
    b = var.abs().mean(dim=1)

    penalty_groups = a[b<thre]
    return penalty_groups.sum(), penalty_groups.numel()

def sparse_loss(autoencoder, images, children):
    loss = 0
    values = images
    model_children = children
    values = F.relu(model_children[0](values))
    loss += torch.mean(torch.abs(values))
    return loss

def img_quant(img):
    image = img
    image[image.le(0.25)]=0
    image[image.ge(0.25)*image.le(1)]=1
    return image 

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def main():
    if args.dataset == "Gen1":
        root="/home2/ahasssan/Prophesee-Gen4/Gen1/"
        classes= 'all'
        #height=720
        #width=1280
        height=240
        width=304
        data='Prophesee'
        if args.event == "histogram":
            event='histogram'
        elif args.event == "event_queue":
            event='event_queue'
        elif args.event == "frame":
            event='frame'
    if args.architecture == "autoencoder":
        train=Autoencoder_training(args, root=root, classes=classes, height=height, width=width, data=data, event=event)
        if args.process == "train":
            train.trainautencoder()
        elif args.process == "test":
            train.testautoencoder()
    if args.architecture == "object-detection":
        if args.process == "train":
            train=DenseObjectDetModel(args, root=root, classes=classes, height=height, width=width, data=data, event=event)
            train.trainEpoch()
        elif args.process == "test":
            test=DenseObjectDetModel(args, root=root, classes=classes, height=height, width=width, data=data, event=event)
            test.testEpoch()

class Autoencoder_training:
    def __init__(self, args, root, classes, height, width, event, data):

        self.train_loader = None
        self.test_loader = None
        self.num_cpu_workers=4
        self.gpu_device=2
        CUDA_VISIBLE_DEVICES=2

        self.model_input_size = torch.tensor([224, 288])
        self.object_classes = classes

        if args.model == "autoencoder":
            self.model=model.QUAutoencoder()
        elif args.model == "dense":
            self.object_det = DenseObjectDet()
        elif args.model == "Yolov2":
            self.object_det =Yolov2_SNN()
        elif args.model == "Yolov5":
            self.object_det =Yolov5()
        self.dataset_path=root
        

        self.steps_lr=[500, 1000, 1500]
        self.factor_lr=0.1

        self.height=height
        self.width=width

        self.nr_events_window=25000
        self.event_representation=event
        self.dataset_name= data

        self.batch_size=5

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
        self.test_loader = self.dataset_loader(test_dataset, batch_size=1,
                                              num_workers=self.num_cpu_workers, pin_memory=False)


    def buildModel(self):
        """Creates the specified model"""
        self.model = self.model
        self.train_loader=self.train_loader
        self.model.cuda()


    def train(self):
        """Main training and validation loop"""

        for i in range(10):
            self.trainautencoder()
            self.epoch_step += 1
            self.scheduler.step()

    def trainautencoder(self):
        self.autoencoder = self.model.cuda()
        self.model = nn.DataParallel(self.model)
        print("Data in Parallel")
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.steps_lr,
                                                                  gamma=self.factor_lr)
        distance = nn.BCELoss()
        #distance = torch.nn.BCEWithLogitsLoss()
        #distance = torch.nn.MSELoss()
        time_step = 9
        epoch = 0
        running_loss=0
        l1=0
        reg_param=0.08
        print(len(self.train_loader))
        for i in range(args.start_epoch, args.epochs):
            for i_batch, sample_batched in enumerate(self.train_loader):
                event, bounding_box, histogram = sample_batched
                self.optimizer.zero_grad()
                histogram = torch.nn.functional.interpolate(histogram.permute(0, 3, 1, 2),
                                                        torch.Size(self.model_input_size))

                Hist=histogram
                '''
                hist = visualizations.visualizeHistogram(Hist[0,:,:,:].permute(1, 2, 0).cpu().detach().numpy())
                plt.imshow(hist)
                #plt.savefig("histogram_prophesee.png")
                plt.savefig("histogram1.png")
                print(np.unique(hist))
                
                '''
                ######### Event queue ############
                ##histogram = histogram.unsqueeze(2)
        
                #output_event=torch.zeros(len(Hist),1,224,288).cuda()
                #output_event=torch.zeros(len(Hist),1,192,256).cuda()
                #l1=0
                #running_loss=0
                print(i_batch)                                        
                ##for step in range(time_step):
                ##step +=1
                x = histogram #[:, step, :, :,:]
                x /= x.max() 
                output = self.autoencoder(x)
                loss = distance(output, x)
                #print(output.unique())
                print(loss)
                print(output.unique())
                l1 += 1
                self.optimizer.zero_grad()
                reg_alpha = torch.tensor(1.).cuda()
                a_lambda = torch.tensor(0.0001).cuda()
                model_children = list(self.autoencoder.children())
                alpha = []
                for name, param in self.autoencoder.named_parameters():
                  if 'alpha' in name:
                    alpha.append(param.item())
                    reg_alpha += param.item() ** 2
                loss += a_lambda * (reg_alpha)
                l1_loss = sparse_loss(self.autoencoder, x, model_children)
                # add the sparsity penalty
                loss = loss + reg_param * l1_loss
                running_loss += loss.item() 
                
                #################################    
                loss.backward()
                self.optimizer.step()
                output= img_quant(output)
                    #print(i_batch)
                    #output_event= torch.cat((output_event, output), 1)
                #loss1 = running_loss/l1
                loss1 = running_loss
                #epoch = epoch+1
                if i_batch % 1000 == 0:
                    hist2 = visualizations.visualizeHistogram(Hist[0,:,:,:].permute(1, 2, 0).cpu().detach().numpy())
                    plt.imshow(hist2)
                    plt.savefig("histogram.png")
                    
                    output = visualizations.visualizeHistogram(output[0,:,:,:].permute(1, 2, 0).cpu().detach().numpy())
                    plt.imshow(output)
                    #plt.savefig("decoder_out_prophesee.png")
                    plt.savefig("decoder_out_4bit.png")
                #save_imgs('Image', x, output)
                #loss1 = running_loss/l1
                #print('epoch {:d}, loss: {:.4f}'.format(epoch + 1, loss1), end='\r')
        torch.save(self.autoencoder.state_dict(), 'autoencoder2_layer_16_Sparse_0.08_new_histogram_4bit_prophesee_gen4.pth')
        self.autoencoder.encoder.register_forward_hook(get_activation('encoder'))
        output = self.autoencoder(x)
        layer_out=activation['encoder']
        #import pdb;pdb.set_trace()
        print("===========")
        count_non=len(torch.nonzero(layer_out.view(-1)))
        print(count_non)
        #print("===========")
        count_t=len(layer_out.view(-1))
        print(count_t)
        out_sparsity = 1-count_non/count_t
        print(out_sparsity)
        print("===========")

## Class for model Inference    
    def testautoencoder(self):
        path = "/home2/ahasssan/Prophesee-Gen4/prophesee_dataloader/autoencoder2_layer_16_Sparse_0.01_new_histogram_4bit_prophesee_gen4.pth"
        self.autoencoder = self.model
        self.autoencoder.load_state_dict(torch.load(path))
        variation=nn.MSELoss()
        Image_diff=torch.Tensor([])
        time_step = 5
        a = 0
        for i_batch, sample_batched in enumerate(self.test_loader):
                event, bounding_box, histogram = sample_batched
                histogram = torch.nn.functional.interpolate(histogram.permute(0, 3, 1, 2),
                                                        torch.Size(self.model_input_size))

                Hist=histogram
                ##histogram = histogram.unsqueeze(2)
                ###histogram=histogram.repeat(1,1,3,1,1)
            
                ##output_event=torch.zeros(len(Hist),1,224,288).cuda()
                #output_event=torch.zeros(len(Hist),1,192,256).cuda()
                #l1=0
                #running_loss=0
                print(i_batch)                                         
                ##for step in range(time_step):
                ##step +=1
                x = histogram #[:, step, :, :,:]
                x /= x.max()
                save_path="/home2/ahasssan/Prophesee-Gen4/prophesee_dataloader/"
                for name, m in self.autoencoder.named_modules():
                    if isinstance(m, QConv2d):
                        m.register_forward_hook(get_activation(name))
                        print("Registered hook for {}".format(name))
                param_path = os.path.join(save_path, "Newparams/Prophesee/POT/0.01")
                if not os.path.isdir(param_path):
                    os.makedirs(param_path)
                count = 0
                output = self.autoencoder(x)
                ############### Data Saving ############
                for name, m in self.autoencoder.named_modules():
                    if isinstance(m, QConv2d):
                        if hasattr(m, 'wint'):
                            if not count in [0]:
                                ae_imgs = self.autoencoder(x)
                                xint = activation[name]
                                input=m.input.data.cpu().numpy()
                                input_re=input.reshape(input.shape[1],input.shape[2]*input.shape[3])
                                input_re=input_re.transpose()

                                wint=m.wint.data.cpu().numpy()
                                wint_re=wint.reshape(wint.shape[0],wint.shape[1]*wint.shape[2]*wint.shape[3])
                                wint_re=wint_re.transpose()
                                ###
                                Xint=m.xint.data.cpu().numpy()
                                Xint_re=Xint.reshape(Xint.shape[1],Xint.shape[2]*Xint.shape[3])
                                Xint_re = Xint_re.transpose()

                                Oint=m.oint.data.cpu().numpy()
                                Oint_re=Oint.reshape(Oint.shape[1],Oint.shape[2]*Oint.shape[3])
                                Oint_re=Oint_re.transpose()
            
                                print(a)
                                print("==========")

                                np.savetxt(param_path+"Input_{}".format(a)+"_unique_{}.csv".format(name),input_re)
                                np.savetxt(param_path+"w_int_{}".format(a)+"_unique_{}.csv".format(name),wint_re)
                                np.savetxt(param_path+"act_int_{}".format(a)+"_unique_{}.csv".format(name),Xint_re)
                                np.savetxt(param_path+"out_int_{}".format(a)+"_unique_{}.csv".format(name),Oint_re)
                                if count==1:
                                    out_1=m.out1.data.cpu().numpy()
                                    out_1=out_1.reshape(out_1.shape[1],out_1.shape[2]*out_1.shape[3])
                                    out_1=out_1.transpose()
                                    np.savetxt(param_path+"maxpool_out_float_{}".format(a)+"_unique_{}.csv".format(name),out_1)
                            count+=1
                output= img_quant(output)
                diff=variation(x,output)
                diff=torch.Tensor([diff])
                print(diff)
                #print(i_batch)
                ##output_event= torch.cat((output_event, output), 1)
                Image_diff= torch.cat((Image_diff,diff), 0)
                if i_batch % 100 ==0:
                    hist = visualizations.visualizeHistogram(Hist[0,:,:,:].permute(1, 2, 0).cpu().detach().numpy())
                    plt.imshow(hist)
                    plt.savefig("Inference_Hist.png")
                    output = visualizations.visualizeHistogram(output[0,:,:,:].permute(1, 2, 0).cpu().detach().numpy())
                    plt.imshow(output)
                    plt.savefig("decoder_Inference_out_4bit_new.png")
        a += 1
        Image_difference = Image_diff.mean()
        print(Image_difference)
class DenseObjectDetModel(Autoencoder_training):
    def __init__(self, args, root, classes, height, width, event, data):

        self.train_loader = None
        self.test_loader = None
        self.num_cpu_workers=4
        self.gpu_device=2
        CUDA_VISIBLE_DEVICES=2

        self.model_input_size = torch.tensor([224, 288])
        self.object_classes = classes

        """Creates the specified model"""
        if args.model == "dense":
            self.object_det = DenseObjectDet()
        elif args.model == "Yolov2":
            self.object_det =Yolov2_SNN()
        elif args.model == "Yolov5":
            self.object_det =Yolov5()
        logger.info(self.object_det)

        self.dataset_path=root
        self.steps_lr=[500, 1000, 1500]
        self.factor_lr=0.1
        self.height=height
        self.width=width
        self.nr_events_window=25000
        self.event_representation=event
        self.dataset_name= data
        self.batch_size=5
        self.dataset_builder = dataset.getDataloader(self.dataset_name)
        self.dataset_loader = Loader
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
        self.test_loader = self.dataset_loader(test_dataset, batch_size=1,
                                              num_workers=self.num_cpu_workers, pin_memory=False)
        
    def saveValidationStatisticsObjectDetection(self):
        """Saves the statistice relevant for object detection"""
        evaluator = VOC_Evaluator()
        metrics = evaluator.GetPascalVOCMetrics(self.bounding_boxes,
                                                IOUThreshold=0.5,
                                                method=MethodAveragePrecision.EveryPointInterpolation)
        acc_AP = 0
        prec=0
        rec=0
        tp=0
        fp=0
        detections=0
        total_positives = 0
        for metricsPerClass in metrics:
            acc_AP += metricsPerClass['AP']
            #prec +=metricsPerClass['interpolated precision']
            #rec +=metricsPerClass['interpolated recall']
            tp +=metricsPerClass['total TP']
            fp += metricsPerClass['total FP']
            total_positives += metricsPerClass['total positives']
            detections += metricsPerClass['total detections']
        mAP = acc_AP / self.nr_classes
        #t_precision= prec/self.nr_classes
        #t_rec=rec/self.nr_classes
        t_TP = tp/self.nr_classes
        t_P = total_positives/ self.nr_classes
        t_FP = fp/self.nr_classes
        t_boxes= detections/self.nr_classes
        print('Total False Positive:{:.10}'.format(t_FP))
        print("================")
        print('Total True Positive:{:.10}'.format(t_TP))
        print("================")
        print('Total Positive:{:.10}'.format(t_P))
        print("================")
        print('Total Events:{:.10}'.format(t_boxes))
        print("================")
        #print('Total Precision:{:.10}'.format(t_precision))
        #print("================")
        #print('Total Recall:{:.10}'.format(t_rec))
        #print("================")
        #import pdb;pdb.set_trace()

        self.validation_accuracy = mAP
        print(self.validation_accuracy)

        return self.validation_accuracy
        #return self.validation_accuracy, t_precision, t_rec
        #self.writer.add_scalar('Validation/Validation_Loss', self.validation_loss, self.epoch_step)
        #self.writer.add_scalar('Validation/Validation_mAP', self.validation_accuracy, self.epoch_step)

    def saveBoundingBoxes(self, gt_bbox, detected_bbox):
        """
        Saves the bounding boxes in the evaluation format
        :param gt_bbox: gt_bbox[0, 0, :]: ['u', 'v', 'w', 'h', 'class_id']
        :param detected_bbox[0, :]: [batch_idx, u, v, w, h, pred_class_id, pred_class_score, object score]
        """
        image_size = self.model_input_size.cpu().numpy()
        for i_batch in range(gt_bbox.shape[0]):
            for i_gt in range(gt_bbox.shape[1]):
                gt_bbox_sample = gt_bbox[i_batch, i_gt, :]
                id_image = 1 * 30 + i_batch
                if gt_bbox[i_batch, i_gt, :].sum() == 0:
                    break

                bb_gt = BoundingBox(id_image, gt_bbox_sample[-1], gt_bbox_sample[0], gt_bbox_sample[1],
                                    gt_bbox_sample[2], gt_bbox_sample[3], image_size, BBType.GroundTruth)
                self.bounding_boxes.addBoundingBox(bb_gt)

        for i_det in range(detected_bbox.shape[0]):
            det_bbox_sample = detected_bbox[i_det, :]
            id_image = 1 * 30 + det_bbox_sample[0]

            bb_det = BoundingBox(id_image, det_bbox_sample[5], det_bbox_sample[1], det_bbox_sample[2],
                                 det_bbox_sample[3], det_bbox_sample[4], image_size, BBType.Detected,
                                 det_bbox_sample[6])
            self.bounding_boxes.addBoundingBox(bb_det)

    def trainEpoch(self):
        self.model = self.object_det.cuda()
        loss_function = yoloLoss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        #self.model_input_size = torch.tensor([223, 287])
        self.model_input_size = torch.tensor([3, 223, 287])
        self.model_input_size_2d = torch.tensor([223, 287])
        #self.model_input_size = torch.tensor([128, 128])
        e=0
        for i in range(args.start_epoch, args.epochs):   
            LOSS = []
            for i_batch, sample_batched in enumerate(self.train_loader):
                event, bounding_box, Histogram = sample_batched
                optimizer.zero_grad()

                ####### Original Version ##################
                histogram = torch.nn.functional.interpolate(Histogram.permute(0, 3, 1, 2),torch.Size(self.model_input_size_2d))
                image = visualizations.visualizeHistogram(histogram[0, :, :, :].permute(1, 2, 0).cpu().int().numpy())
                ##### For Input size 2D
                # bounding_box[:, :, [0, 2]] = (bounding_box[:, :, [0, 2]] * self.model_input_size[1].float()
                #                             / 304).long()
                # bounding_box[:, :, [1, 3]] = (bounding_box[:, :, [1, 3]] * self.model_input_size[0].float()
                #                             / 240).long()
                
                bounding_box[:, :, [0, 2]] = (bounding_box[:, :, [0, 2]] * self.model_input_size[2].float()
                                            / 304).long()
                bounding_box[:, :, [1, 3]] = (bounding_box[:, :, [1, 3]] * self.model_input_size[1].float()
                                            / 240).long()
                
                bounding_box[:, :, [0, 2]] = (bounding_box[:, :, [0, 2]]).long()
                bounding_box[:, :, [1, 3]] = (bounding_box[:, :, [1, 3]]).long()

                histogram = histogram.unsqueeze_(1)
                histogram = histogram.float()
                self.model_input_size_2d = torch.tensor([223, 287])
                model_output = self.model(histogram)
                out = loss_function(model_output, bounding_box, self.model_input_size_2d)
                loss = out[0]
                loss.backward()
                optimizer.step()

                if i_batch%args.frequency == 0:
                    with torch.no_grad():
                       detected_bbox = yoloDetect(model_output, self.model_input_size_2d.to(model_output.device),
                                                   threshold=0.5).long().cpu().numpy()
                       detected_bbox = detected_bbox[detected_bbox[:, 0] == 0, 1:-2]

                    #image = visualizations.visualizeHistogram(histogram[0, :, :, :].permute(1, 2, 0).cpu().int().numpy())
                    image = visualizations.visualizeHistogram(histogram.squeeze_(1)[0, :, :, :].permute(1, 2, 0).cpu().int().numpy())
                    image = visualizations.drawBoundingBoxes(image, bounding_box[0, :, :].cpu().numpy(),
                                                                class_name=[self.object_classes[i]
                                                                            for i in bounding_box[0, :, -1]],
                                                                ground_truth=True, rescale_image=True)
                    image = visualizations.drawBoundingBoxes(image, detected_bbox[:, :-1],
                                                               class_name=[self.object_classes[i]
                                                                           for i in detected_bbox[:, -1]],
                                                               ground_truth=False, rescale_image=False)
                    plt.imshow(image)
                    plt.savefig(args.save_images)    
                LOSS.append(loss.data.cpu().numpy())
                # print(loss.data.cpu().numpy())
                # print("=====================")
            print(i)
            print("loss = ", np.mean(LOSS))
            torch.save(self.model.state_dict(), args.save_path)

    def testEpoch(self):
        self.autoencoder=model.QUAutoencoder().cuda()
        #self.object_det = DenseObjectDet()
        self.model = self.object_det.cuda()
        path1="/home2/ahasssan/Prophesee-Gen4/prophesee_dataloader/autoencoder2_layer_16_Sparse_0.05_new_histogram_4bit_prophesee_gen4.pth"
        self.autoencoder.load_state_dict(torch.load(path1))

        path2=args.save_path
        self.model.load_state_dict(torch.load(path2))
        self.model_input_size_2d = torch.tensor([223, 287])
        self.model_input_size = torch.tensor([3,223, 287])
        self.bounding_boxes = BoundingBoxes()
        loss_function = yoloLoss
        # Images are upsampled for visualization
        val_images = np.zeros([2, int(self.model_input_size_2d[0]*1.5), int(self.model_input_size_2d[1]*1.5), 3])

        for i_batch, sample_batched in enumerate(self.test_loader):
            event, bounding_box, Histogram = sample_batched
            # Convert spatial dimension to model input size
            histogram = torch.nn.functional.interpolate(Histogram.permute(0, 3, 1, 2),
                                                        torch.Size(self.model_input_size_2d))
            # histogram = torch.nn.functional.interpolate(Histogram.permute(0, 1, 4, 2, 3),
            #                                                 torch.Size(self.model_input_size))
            # x = histogram #[:, step, :, :,:]
            # x /= x.max()
            # output = self.autoencoder(x)
            # histogram= img_quant(output)                                            
            # Change x, width and y, height
            histogram = histogram.unsqueeze_(1)
            histogram = histogram.float()
            bounding_box[:, :, [0, 2]] = (bounding_box[:, :, [0, 2]] * self.model_input_size[2].float()
                                          / 304).long()
            bounding_box[:, :, [1, 3]] = (bounding_box[:, :, [1, 3]] * self.model_input_size[1].float()
                                          / 240).long()
            bounding_box[:, :, [0, 2]] = (bounding_box[:, :, [0, 2]]).long()
            bounding_box[:, :, [1, 3]] = (bounding_box[:, :, [1, 3]]).long()


            with torch.no_grad():
                #histogram = histogram.unsqueeze_(1)
                cnt = 0
                for m in self.model.modules():
                    if isinstance(m, ZIFArchTan):
                        print(m)
                        m.neuron_idx = cnt
                        cnt += 1
                model_output = self.model(histogram)
                import pdb;pdb.set_trace()
                loss = loss_function(model_output, bounding_box, self.model_input_size_2d)[0]
                detected_bbox = yoloDetect(model_output, self.model_input_size_2d.to(model_output.device),
                                           threshold=0.4)
                detected_bbox = nonMaxSuppression(detected_bbox, iou=0.5)
                detected_bbox = detected_bbox.cpu().numpy()
            # Save validation statistics
            self.saveBoundingBoxes(bounding_box.cpu().numpy(), detected_bbox)

            #if self.val_batch_step % (self.nr_val_epochs - 2) == 0:
            vis_detected_bbox = detected_bbox[detected_bbox[:, 0] == 0, 1:-2].astype(np.int)

            image = visualizations.visualizeHistogram(histogram.squeeze_(1)[0, :, :, :].permute(1, 2, 0).cpu().int().numpy())
            #image = visualizations.visualizeHistogram(histogram[0, :, :, :].permute(1, 2, 0).cpu().int().numpy())
            #plt.imshow(image)
            #plt.savefig("histogram.png")
            image = visualizations.drawBoundingBoxes(image, bounding_box[0, :, :].cpu().numpy(),
                                                         class_name=[self.object_classes[i]
                                                                     for i in bounding_box[0, :, -1]],
                                                         ground_truth=True, rescale_image=True)
            image = visualizations.drawBoundingBoxes(image, vis_detected_bbox[:, :-1],
                                                         class_name=[self.object_classes[i]
                                                                     for i in vis_detected_bbox[:, -1]],
                                                         ground_truth=False, rescale_image=False)
            #import pdb; pdb.set_trace()
            #val_images[int(self.val_batch_step // (self.nr_val_epochs - 2))] = image

            print('Valid_loss:{:.8}'.format(loss.data.cpu().numpy()))
            #print(ValLoss=loss.data.cpu().numpy())

        plt.imshow(image)
        plt.savefig("Inference_image_boundingboxes.png")
        #self.saveValidationStatisticsObjectDetection()
        #self.writer.add_image('Validation/Input Histogram', val_images, self.epoch_step, dataformats='NHWC')
        mean_AP = self.saveValidationStatisticsObjectDetection()
        #mean_AP = params[0]
        #Precision = params[1]
        #Recall = params[2]
        #Tp= params[3]
        #Fp= params[4]

        print("========")
        print('Mean Average Precision:{:.10}'.format(mean_AP))
        #print('Final Precision:{:.10}'.format(precision))
        #print('Final Recall:{:.10}'.format(Recall))
        #print('Total True Positive:{:.10}'.format(Tp))
        #print('Total False Positive:{:.10}'.format(Fp))




if __name__=='__main__':
    main()