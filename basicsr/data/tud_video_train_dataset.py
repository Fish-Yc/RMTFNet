import numpy as np
import random
import cv2
import torch
from pathlib import Path
import torch.utils.data as data
from torchvision import transforms

from basicsr.data.transforms import paired_random_crop, augment
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

def read_img(path):
    # read image by cv2
    # return: Numpy float32, HWC, BGR, [0,1]
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # cv2.IMREAD_GRAYSCALE
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img

@DATASET_REGISTRY.register()
class TUDVideoDataset(data.Dataset):
    """TUD dataset for Training
    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            dataroot_flow (str, optional): Data root path for flow.
            meta_info_file (str): Path for meta information file.
            val_partition (str): Validation partition types. 'REDS4' or
                'official'.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            interval_list (list): Interval list for temporal augmentation.
            random_reverse (bool): Random reverse input frames.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(TUDVideoDataset, self).__init__()
        self.opt = opt
        self.scale = opt['scale']
        self.gt_size = opt['gt_size']
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.num_frame = opt['num_frame']
        self.filename_tmpl = opt['filename_tmpl']
        self.filename_ext = opt['filename_ext']
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        self.use_hflip = opt.get('use_hflip', True)
        self.use_rot = opt.get('use_rot', True)
        

        keys = []
        total_num_frames = [] # some clips may not have 100 frames
        start_frames = [] # some clips may not start from 00000
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                folder, frame_num, _, start_frame = line.split()
                frame_num = int(frame_num)
                start_frame = int(start_frame)
                keys.extend([f'{folder}/{i:{self.filename_tmpl}}' for i in range(start_frame, start_frame + frame_num)])
                total_num_frames.extend([frame_num for i in range(frame_num)])
                start_frames.extend([start_frame for i in range(frame_num)])

        val_partition = []

        self.keys = []
        self.total_num_frames = [] # some clips may not have 100 frames
        self.start_frames = []
        for i, v in zip(range(len(keys)), keys):
            if v.split('/')[0] not in val_partition:
                self.keys.append(keys[i])
                self.total_num_frames.append(total_num_frames[i])
                self.start_frames.append(start_frames[i])


    def __getitem__(self, index):
        
        key = self.keys[index]
        total_num_frames = self.total_num_frames[index]
        start_frames = self.start_frames[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000

        # determine the neighboring frames
        valid_intervals = [
            interval for interval in self.interval_list
            if total_num_frames >= 1 + (self.num_frame - 1) * interval
        ]
        interval = random.choice(valid_intervals) if valid_intervals else 1

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        endmost_start_frame_idx = start_frames + total_num_frames - 1 - (self.num_frame - 1) * interval
        if start_frame_idx > endmost_start_frame_idx:
            start_frame_idx = random.randint(start_frames, endmost_start_frame_idx)
        end_frame_idx = start_frame_idx + self.num_frame * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()


        # get the neighboring LQ and GT frames
        img_lqs = []
        img_gts = []
        for neighbor in neighbor_list:
            img_lq_path = self.lq_root / clip_name / f'{neighbor:{self.filename_tmpl}}.{self.filename_ext}'
            img_gt_path = self.gt_root / clip_name / f'{neighbor:{self.filename_tmpl}}.{self.filename_ext}'

            # get LQ
            img_lq = read_img(img_lq_path)
            img_lqs.append(img_lq)

            # get GT
            img_gt = read_img(img_gt_path)
            img_gts.append(img_gt)

        # randomly crop
        img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, self.gt_size, self.scale, img_gt_path)

        # augmentation - flip, rotate
        img_results = augment(img_lqs + img_gts, hflip=self.use_hflip, rotation=self.use_rot)

        img_results = img2tensor(img_results)
        img_lqs = torch.stack(img_results[:len(neighbor_list)], dim=0)
        img_gts = torch.stack(img_results[len(neighbor_list):], dim=0)


        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        
        return {'lq': img_lqs, 'gt': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys)
