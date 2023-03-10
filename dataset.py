import os
import tqdm
import random
import numpy as np
import torch
from os import listdir
from os.path import join
import event_representations as er
from numpy.lib import recfunctions as rfn
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from prophesee import dat_events_tools
from prophesee import npy_events_tools
import read_events

RED = np.array(((1, 0, 0)), dtype=np.uint8)
GREEN = np.array(((0, 1, 0)), dtype=np.uint8)
WHITE = np.array(((255, 255, 255)), dtype=np.uint8)
BLACK = np.array(((0, 0, 0)), dtype=np.uint8)
GREY = np.array(((220, 220, 220)), dtype=np.uint8)

def random_shift_events(events, max_shift=20, resolution=(180, 240), bounding_box=None):
    H, W = resolution
    if bounding_box is not None:
        x_shift = np.random.randint(-min(bounding_box[0, 0], max_shift),
                                    min(W - bounding_box[2, 0], max_shift), size=(1,))
        y_shift = np.random.randint(-min(bounding_box[0, 1], max_shift),
                                    min(H - bounding_box[2, 1], max_shift), size=(1,))
        bounding_box[:, 0] += x_shift
        bounding_box[:, 1] += y_shift
    else:
        x_shift, y_shift = np.random.randint(-max_shift, max_shift+1, size=(2,))

    events[:, 0] += x_shift
    events[:, 1] += y_shift

    valid_events = (events[:, 0] >= 0) & (events[:, 0] < W) & (events[:, 1] >= 0) & (events[:, 1] < H)
    events = events[valid_events]

    if bounding_box is None:
        return events

    return events, bounding_box


def random_flip_events_along_x(events, resolution=(180, 240), p=0.5, bounding_box=None):
    H, W = resolution
    flipped = False
    if np.random.random() < p:
        events[:, 0] = W - 1 - events[:, 0]
        flipped = True

    if bounding_box is None:
        return events

    if flipped:
        bounding_box[:, 0] = W - 1 - bounding_box[:, 0]
        bounding_box = bounding_box[[1, 0, 3, 2]]
    return events, bounding_box


def getDataloader(name):
    dataset_dict = {'NCaltech101': NCaltech101,
                    'NCaltech101_ObjectDetection': NCaltech101_ObjectDetection,
                    'Prophesee': Prophesee,
                    'NCars': NCars}
    return dataset_dict.get(name)


class NCaltech101:
    def __init__(self, root, object_classes, height, width, nr_events_window=-1, augmentation=False, mode='training',
                 event_representation='histogram', shuffle=True):
        if mode == 'training':
            mode = 'training'
        elif mode == 'testing':
            mode = 'testing'
        if mode == 'validation':
            mode = 'valdation'
        root = os.path.join(root, mode)

        if object_classes == 'all':
            self.object_classes = listdir(root)
        else:
            self.object_classes = object_classes

        self.width = width
        self.height = height
        self.augmentation = augmentation
        self.nr_events_window = nr_events_window
        self.nr_classes = len(self.object_classes)
        self.event_representation = event_representation

        self.files = []
        self.labels = []

        for i, object_class in enumerate(self.object_classes):
            new_files = [join(root, object_class, f) for f in listdir(join(root, object_class))]
            self.files += new_files
            self.labels += [i] * len(new_files)

        self.nr_samples = len(self.labels)

        if shuffle:
            zipped_lists = list(zip(self.files, self.labels))
            random.seed(7)
            random.shuffle(zipped_lists)
            self.files, self.labels = zip(*zipped_lists)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        label = self.labels[idx]
        filename = self.files[idx]
        events = np.load(filename).astype(np.float32)
        nr_events = events.shape[0]

        window_start = 0
        window_end = nr_events
        if self.augmentation:
            events = random_shift_events(events)
            events = random_flip_events_along_x(events)
            window_start = random.randrange(0, max(1, nr_events - self.nr_events_window))

        if self.nr_events_window != -1:
            # Catch case if number of events in batch is lower than number of events in window.
            window_end = min(nr_events, window_start + self.nr_events_window)

        events = events[window_start:window_end, :]

        histogram = self.generate_input_representation(events, (self.height, self.width))
        
        #import pdb;pdb.set_trace()
        return events, label, histogram

    def generate_input_representation(self, events, shape):
        """
        Events: N x 4, where cols are x, y, t, polarity, and polarity is in {0,1}. x and y correspond to image
        coordinates u and v.
        """
        if self.event_representation == 'histogram':
            return self.generate_event_histogram(events, shape)
            #return self.get_frames(events, shape)
        elif self.event_representation == 'event_queue':
            return self.generate_event_queue(events, shape)
        elif self.event_representation == 'frames':
            return self.get_frames(events, shape)

    @staticmethod
    def generate_event_histogram(events, shape):
        """
        Events: N x 4, where cols are x, y, t, polarity, and polarity is in {0,1}. x and y correspond to image
        coordinates u and v.
        """
        H, W = shape
        x, y, t, p = events.T
        x = x.astype(np.int)
        y = y.astype(np.int)

        img_pos = np.zeros((H * W,), dtype="float32")
        img_neg = np.zeros((H * W,), dtype="float32")
        
        #frame_data[(i, y, x)] = colors
        ##np.add.at(img_pos, x[p == 1] + W * y[p == 1], 1)
        np.put(img_pos, x[p == 1] + W * y[p == 1], 1)
        ##np.add.at(img_neg, x[p == -1] + W * y[p == -1], 1)
        np.put(img_neg, x[p == -1] + W * y[p == -1], 1)
        #histogram = img_pos.reshape(H,W)
        histogram = np.stack([img_neg, img_pos], -1).reshape((H, W, 2))

        return histogram
    
    @staticmethod
    def generate_event_histogram_timestamped(events, shape, polarity=None, dt=None, num_frames=4, flip_up_down=False):
        """
        Events: N x 4, where cols are x, y, t, polarity, and polarity is in {0,1}. x and y correspond to image
        coordinates u and v.
        """
        H, W = shape
        x, y, t, p = events.T
        time = t
        x = x.astype(np.int)
        y = y.astype(np.int)

        assert time.size > 0, "the length of the time sequence must greater than 0!"
        t_start = time[0]
        t_end = time[-1]
        if dt is None:
            dt = int((t_end - t_start) // (num_frames - 1))
        else:
            num_frames = (t_end - t_start) // dt + 1

        img_pos = np.zeros((num_frames,H * W,1), dtype="float32")
        img_neg = np.zeros((num_frames,H * W,1), dtype="float32")
        #frame_data[(i, y, x)] = colors
        ##np.add.at(img_pos, x[p == 1] + W * y[p == 1], 1)
        np.put(img_pos, x[p == 1] + W * y[p == 1], 1)
        ##np.add.at(img_neg, x[p == -1] + W * y[p == -1], 1)
        np.put(img_neg, x[p == -1] + W * y[p == -1], 1)
        #histogram = img_pos.reshape(H,W)
        i = np.minimum((time-t_start) // dt, num_frames - 1)
        img_pos[(i, y, x)] = img_pos
        img_neg[(i, y, x)] = img_neg
        histogram = np.stack([img_neg, img_pos], -1).reshape((4,H, W, 2))

        return histogram

    @staticmethod
    def get_frames(
        events, shape,
        polarity=None,
        dt=None,
        num_frames=3,
        flip_up_down=False
    ):
        r"""
        convert the events to rgb frames
        """
        H, W = shape
        x, y, time, p = events.T
        coords = np.stack((x, y), axis=-1)
        polarity=p
        x = x.astype(np.int)
        y = y.astype(np.int)

        assert time.size > 0, "the length of the time sequence must greater than 0!"
        t_start = time[0]
        t_end = time[-1]
        if dt is None:
            dt = int((t_end - t_start) // (num_frames - 1))
        else:
            num_frames = (t_end - t_start) // dt + 1

        frame_data1 = np.zeros((num_frames, *shape, 1), dtype=np.uint8)
        frame_data2 = np.zeros((num_frames, *shape, 1), dtype=np.uint8)
        if polarity is None:
            colors1 = GREY
        else:
            colors1 = np.where(polarity[:, np.newaxis]==1, RED, GREEN)
            #colors1 = np.put(frame_data1, x[p == 1] + W * y[p == 1], 1)
            colors2 = np.where(polarity[:, np.newaxis]==-1, RED, GREEN)
            #colors2 = np.put(frame_data2, x[p == -1] + W * y[p == -1], 1)

        i = np.minimum((time-t_start) // dt, num_frames - 1)
        x, y = coords.T
        if flip_up_down:
            y = shape[0] - y - 1
        frame_data1[(i, y, x)] = colors1[:,:1]
        frame_data2[(i, y, x)] = colors2[:,:1]
        frame_data = np.stack([frame_data1, frame_data2], -1).reshape((3,H, W, 2))
        frame_data = frame_data[:2,:,:,:]
        return frame_data

    @staticmethod
    def generate_event_queue(events, shape, K=8):
        """
        Events: N x 4, where cols are x, y, t, polarity, and polarity is in {0,1}. x and y correspond to image
        coordinates u and v.
        """
        H, W = shape
        events = events.astype(np.float32)
        events[events<0]=1
        if events.shape[0] == 0:
            return np.zeros([H, W, 2*K], dtype=np.float32)

        # [2, K, height, width],  [0, ...] time, [:, 0, :, :] newest events
        four_d_tensor = er.event_queue_tensor(events, K, H, W, -1).astype(np.float32)

        # Normalize
        four_d_tensor[0, ...] = four_d_tensor[0, 0, None, :, :] - four_d_tensor[0, :, :, :]
        max_timestep = np.amax(four_d_tensor[0, :, :, :], axis=0, keepdims=True)

        # four_d_tensor[0, ...] = np.divide(four_d_tensor[0, ...], max_timestep, where=max_timestep.astype(np.bool))
        four_d_tensor[0, ...] = four_d_tensor[0, ...] / (max_timestep + (max_timestep == 0).astype(np.float))
        #four_d_tensor = four_d_tensor.cpu().int().numpy()
        #four_d_tensor=np.expand_dims(four_d_tensor, axis=1)
        #four_d_tensor=torch.tensor(four_d_tensor)
        four_d_tensor=four_d_tensor.reshape([2*K, H, W]).transpose(1, 2, 0)
        #four_d_tensor= cv2.cvtColor(four_d_tensor, cv2.COLOR_GRAY2BGR)
        #import pdb;pdb.set_trace()
        return four_d_tensor #.reshape([2*K, H, W]).transpose(1, 2, 0)  

class Prophesee(NCaltech101):
    def __init__(self, root, object_classes, height, width, nr_events_window=-1, augmentation=False, mode='training',
                 event_representation='histogram', shuffle=True):
        
        if mode == 'training':
            mode = 'train'
        elif mode == 'validation':
            mode = 'valid'
        elif mode == 'testing':
            mode = 'test'

        #file_dir = os.path.join('detection_dataset_duration_60s_ratio_1.0', mode)
        file_dir = mode
        self.files = listdir(os.path.join(root, mode))
        # Remove duplicates (.npy and .dat)
        self.files = [os.path.join(file_dir, time_seq_name[:-9]) for time_seq_name in self.files
                      if time_seq_name[-3:] == 'npy']

        self.root = root
        self.mode = mode
        self.width = width
        self.height = height
        self.augmentation = augmentation
        self.event_representation = event_representation
        if nr_events_window == -1:
            self.nr_events_window = 250000
        else:
            self.nr_events_window = nr_events_window

        self.max_nr_bbox = 45

        if object_classes == 'all':
            #self.nr_classes = 7
            self.nr_classes = 2
            self.object_classes = ['Car', "Pedestrian"]
            #self.object_classes = ["pedestrian", "two wheeler", "car", "truck", "bus", "traffic sign", "traffic light"]
        else:
            self.nr_classes = len(object_classes)
            self.object_classes = object_classes

        self.sequence_start = []
        self.createAllBBoxDataset()
        self.nr_samples = len(self.files)

        if shuffle:
            zipped_lists = list(zip(self.files,  self.sequence_start))
            random.seed(7)
            random.shuffle(zipped_lists)
            self.files,  self.sequence_start = zip(*zipped_lists)

    def createAllBBoxDataset(self):
        """
        Iterates over the files and stores for each unique bounding box timestep the file name and the index of the
         unique indices file.
        """
        file_name_bbox_id = []
        print('Building the Dataset')
        pbar = tqdm.tqdm(total=len(self.files), unit='File', unit_scale=True)

        for i_file, file_name in enumerate(self.files):
            bbox_file = os.path.join(self.root, file_name + '_bbox.npy')
            event_file = os.path.join(self.root, file_name + '_td.dat')
            f_bbox = open(bbox_file, "rb")
            start, v_type, ev_size, size = npy_events_tools.parse_header(f_bbox)
            dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
            f_bbox.close()
            #### Change here####
            ####################
            unique_ts, unique_indices = np.unique(dat_bbox['ts'], return_index=True)

            for unique_time in unique_ts:
                sequence_start = self.searchEventSequence(event_file, unique_time, nr_window_events=250000)
                self.sequence_start.append(sequence_start)

            file_name_bbox_id += [[file_name, i] for i in range(len(unique_indices))]
            pbar.update(1)

        pbar.close()
        self.files = file_name_bbox_id

    def __getitem__(self, idx):
        bbox_file = os.path.join(self.root, self.files[idx][0] + '_bbox.npy')
        event_file = os.path.join(self.root, self.files[idx][0] + '_td.dat')

        # Bounding Box
        f_bbox = open(bbox_file, "rb")
        # dat_bbox types (v_type):
        # [('ts', 'uint64'), ('x', 'float32'), ('y', 'float32'), ('w', 'float32'), ('h', 'float32'), (
        # 'class_id', 'uint8'), ('confidence', 'float32'), ('track_id', 'uint32')]
        start, v_type, ev_size, size = npy_events_tools.parse_header(f_bbox)
        dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
        f_bbox.close()

        #### Change here####
        ####################
        unique_ts, unique_indices = np.unique(dat_bbox['ts'], return_index=True)
        nr_unique_ts = unique_ts.shape[0]

        bbox_time_idx = self.files[idx][1]

        # Get bounding boxes at current timestep
        #### Change here####
        ####################
        if bbox_time_idx == (nr_unique_ts - 1):
            end_idx = dat_bbox['ts'].shape[0]
        else:
            end_idx = unique_indices[bbox_time_idx+1]

        bboxes = dat_bbox[unique_indices[bbox_time_idx]:end_idx]

        # Required Information ['x', 'y', 'w', 'h', 'class_id']
        np_bbox = rfn.structured_to_unstructured(bboxes)[:, [1, 2, 3, 4, 5]]
        np_bbox = self.cropToFrame(np_bbox)

        const_size_bbox = np.zeros([self.max_nr_bbox, 5])
        const_size_bbox[:np_bbox.shape[0], :] = np_bbox

        # Events
        events = self.readEventFile(event_file, self.sequence_start[idx],  nr_window_events=self.nr_events_window)
        histogram = self.generate_input_representation(events, (self.height, self.width))
        return events, const_size_bbox.astype(np.int64), histogram

    def searchEventSequence(self, event_file, bbox_time, nr_window_events=250000):
        term_criterion = nr_window_events // 2
        nr_events = dat_events_tools.count_events(event_file)
        file_handle = open(event_file, "rb")
        ev_start, ev_type, ev_size, img_size = dat_events_tools.parse_header(file_handle)
        low = 0
        high = nr_events

        # binary search
        while high - low > term_criterion:
            middle = (low + high) // 2

            # self.seek_event(file_handle, middle)
            file_handle.seek(ev_start + middle * ev_size)
            mid = np.fromfile(file_handle, dtype=[('ts', 'u4'), ('_', 'i4')], count=1)["ts"][0]

            if mid > bbox_time:
                high = middle
            elif mid < bbox_time:
                low = middle + 1
            else:
                file_handle.seek(ev_start + (middle - (term_criterion // 2) * ev_size))
                break

        file_handle.close()
        # we now know that it is between low and high
        return ev_start + low * ev_size

    def readEventFile(self, event_file, file_position, nr_window_events=250000):
        file_handle = open(event_file, "rb")
        # file_position = ev_start + low * ev_size
        file_handle.seek(file_position)
        dat_event = np.fromfile(file_handle, dtype=[('ts', 'u4'), ('_', 'i4')], count=nr_window_events)
        file_handle.close()

        x = np.bitwise_and(dat_event["_"], 16383)
        y = np.right_shift(
            np.bitwise_and(dat_event["_"], 268419072), 14)
        p = np.right_shift(np.bitwise_and(dat_event["_"], 268435456), 28)
        p[p == 0] = -1
        events_np = np.stack([x, y, dat_event['ts'], p], axis=-1)

        return events_np

    def cropToFrame(self, np_bbox):
        """Checks if bounding boxes are inside frame. If not crop to border"""
        array_width = np.ones_like(np_bbox[:, 0]) * self.width - 1
        array_height = np.ones_like(np_bbox[:, 1]) * self.height - 1

        np_bbox[:, :2] = np.maximum(np_bbox[:, :2], np.zeros_like(np_bbox[:, :2]))
        np_bbox[:, 0] = np.minimum(np_bbox[:, 0], array_width)
        np_bbox[:, 1] = np.minimum(np_bbox[:, 1], array_height)

        np_bbox[:, 2] = np.minimum(np_bbox[:, 2], array_width - np_bbox[:, 0])
        np_bbox[:, 3] = np.minimum(np_bbox[:, 3], array_height - np_bbox[:, 1])

        return np_bbox
