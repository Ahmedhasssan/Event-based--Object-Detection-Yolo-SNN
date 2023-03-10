from os.path import join
import sys
import argparse
import numpy as np
import visualizations as visualizations

# comet_available = False
try:
    import comet_ml
    # comet_available = True
except ImportError:
    print("Comet is not installed, Comet logger will not be available.")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger

import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import cv2
import sys

from gen1_od_dataset import GEN1DetectionDataset

from torch.utils.data.dataloader import default_collate

parser = argparse.ArgumentParser(description='Classify event dataset')
# Dataset
parser.add_argument('-dataset', default='gen1', type=str, help='dataset used {GEN1}')
parser.add_argument('-path', default='PropheseeGEN1', type=str, help='path to dataset location')
parser.add_argument('-num_classes', default=2, type=int, help='number of classes')

# Data
parser.add_argument('-b', default=64, type=int, help='batch size')
parser.add_argument('-sample_size', default=100000, type=int, help='duration of a sample in µs')
parser.add_argument('-T', default=5, type=int, help='simulating time-steps')
parser.add_argument('-tbin', default=2, type=int, help='number of micro time bins')
parser.add_argument('-image_shape', default=(240,304), type=tuple, help='spatial resolution of events')

# Training
parser.add_argument('-epochs', default=50, type=int, help='number of total epochs to run')
parser.add_argument('-lr', default=1e-3, type=float, help='learning rate used')
parser.add_argument('-wd', default=1e-4, type=float, help='weight decay used')
parser.add_argument('-num_workers', default=4, type=int, help='number of workers for dataloaders')
parser.add_argument('-no_train', action='store_false', help='whether to train the model', dest='train')
parser.add_argument('-test', action='store_true', help='whether to test the model')
parser.add_argument('-device', default=0, type=int, help='device')
parser.add_argument('-precision', default=16, type=int, help='whether to use AMP {16, 32, 64}')
parser.add_argument('-save_ckpt', action='store_true', help='saves checkpoints')
parser.add_argument('-comet_api', default=None, type=str, help='api key for Comet Logger')

# Backbone
parser.add_argument('-backbone', default='vgg-11', type=str, help='model used {squeezenet-v, vgg-v, mobilenet-v, densenet-v}', dest='model')
parser.add_argument('-cfg', type=str, help='configuration of the layers of a custom VGG')
parser.add_argument('-no_bn', action='store_false', help='don\'t use BatchNorm2d', dest='bn')
parser.add_argument('-pretrained_backbone', default=None, type=str, help='path to pretrained backbone model')
parser.add_argument('-pretrained', default=None, type=str, help='path to pretrained model')
parser.add_argument('-extras', type=int, default=[640, 320, 320], nargs=4, help='number of channels for extra layers after the backbone')


# Priors
parser.add_argument('-min_ratio', default=0.05, type=float, help='min ratio for priors\' box generation')
parser.add_argument('-max_ratio', default=0.80, type=float, help='max ratio for priors\' box generation')
parser.add_argument('-aspect_ratios', default=[[2], [2, 3], [2, 3], [2, 3], [2], [2]], type=int, help='aspect ratios for priors\' box generation')

# Loss parameters
parser.add_argument('-box_coder_weights', default=[10.0, 10.0, 5.0, 5.0], type=float, nargs=4, help='weights for the BoxCoder class')
parser.add_argument('-iou_threshold', default=0.50, type=float, help='intersection over union threshold for the SSDMatcher class')
parser.add_argument('-score_thresh', default=0.01, type=float, help='score threshold used for postprocessing the detections')
parser.add_argument('-nms_thresh', default=0.45, type=float, help='NMS threshold used for postprocessing the detections')
parser.add_argument('-topk_candidates', default=200, type=int, help='number of best detections to keep before NMS')
parser.add_argument('-detections_per_img', default=100, type=int, help='number of best detections to keep after NMS')

args = parser.parse_args()
if args.dataset == "gen1":
        dataset = GEN1DetectionDataset
else:
        sys.exit(f"{args.dataset} is not a supported dataset.")

def drawBoundingBoxes(imageData, imageOutputPath, inferenceResults, color):
    """Draw bounding boxes on an image.
    imageData: image data in numpy array format
    imageOutputPath: output image file path
    inferenceResults: inference results array off object (l,t,w,h)
    colorMap: Bounding box color candidates, list of RGB tuples.
    """
    bbox=inferenceResults['boxes']
    label = inferenceResults['labels']
    for i,res in enumerate(bbox):
        left = int(res[0])
        top = int(res[1])
        right = int(res[2])
        bottom = int(res[3])
        # right = int(res[0]) + int(res[2])
        # bottom = int(res[1]) + int(res[3])
        if i<len(label):
            target = label[i]
        else:
            target = label
        if target==0:
            target = "Car"
        else:
            target = "Padestrain"
        new_dim = np.array(imageData.shape[:2], dtype=np.float)
        imageData = cv2.resize(imageData, tuple(new_dim.astype(int)[::-1]), interpolation=cv2.INTER_NEAREST)
        imgHeight, imgWidth, _ = imageData.shape
        thick = int((imgHeight + imgWidth) // 300)
        print (left, top, right, bottom)
        cv2.rectangle(imageData,(left, top), (right, bottom), color, thick)
        cv2.putText(imageData, target, (left, top - 12), 0, 0.2, color, thick//3, cv2.LINE_AA)
    return imageData
    #cv2.imwrite(imageOutputPath, imageData)

def collate_events(data):
    labels = []
    histograms = []
    for i, d in enumerate(data):
        labels.append(d[1])
        histograms.append(d[0])
    labels = default_collate(labels)
    histograms = default_collate(histograms)

    return histograms, labels

def collate_fn(batch):
    targets = torch.zeros(5)
    labels = torch.Tensor([])
    samples = [item[0] for item in batch]
    samples = torch.stack(samples, 0)
    for i, d in enumerate(batch):
        labels=torch.cat((labels,d[1]),0)
    target = [item[1] for item in batch]
    return [samples, target]
train_dataset = dataset(args, mode="train")
#val_dataset = dataset(args, mode="val") 
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.b, collate_fn=collate_fn, 
                                                      num_workers=args.num_workers, pin_memory=False)
#train_dataloader = DataLoader(train_dataset, batch_size=args.b, num_workers=args.num_workers, pin_memory=False)
model_input_size = torch.tensor([223, 287])
for i_batch, sample_batched in enumerate(train_dataloader):
#for sample_batched in train_dataloader:
            Histogram, bounding_box = sample_batched
            import pdb;pdb.set_trace()
            for h in range(4):
                histogram = Histogram[:,h,:,:,:]
                #histogram = Histogram
                histogram = torch.nn.functional.interpolate(histogram.permute(0, 1, 2, 3),
                                                                   torch.Size(model_input_size))
                #histogram = histogram.permute(0, 3, 1, 2)
                image = histogram[0, :, :, :].permute(1, 2, 0).cpu().int().numpy()
                #image = histogram[0, :, :, :].permute(1, 2, 0)
                image = image/image.max()
                image = image[:, :, :3]
                color = (0, 1, 1)
                image = drawBoundingBoxes(image, './output.png', bounding_box[0], color)
                plt.imshow(image)
                plt.savefig("images"+ h +".png")
                #import pdb;pdb.set_trace()
                # image = visualizations.visualizeHistogram(histogram[0, :, :, :].permute(1, 2, 0).cpu().int().numpy())
                # image = image[:, :, :3]
                # # print(np.unique(image))
                # bounding_box[:, :, [0, 2]] = (bounding_box[:, :, [0, 2]] * model_input_size[1].float()
                #                             / 304).long()
                # bounding_box[:, :, [1, 3]] = (bounding_box[:, :, [1, 3]] * model_input_size[0].float()
                #                             / 240).long()


                # image = visualizations.visualizeHistogram(histogram[0, :, :, :].permute(1, 2, 0).cpu().int().numpy())
                # image = image[:, :, :3]
                # image = visualizations.drawBoundingBoxes(image, bounding_box[0, :, :].cpu().numpy(),
                #                                             class_name=[num_classes[i]
                #                                                         for i in bounding_box[0, :, -1]],
                #                                             ground_truth=True, rescale_image=True)
                # plt.imshow(image)
                # plt.savefig("images.png")
                #print(i_batch)
