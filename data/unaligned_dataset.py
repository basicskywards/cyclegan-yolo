import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
from pdb import set_trace as st
import torch
import numpy as np 
#from yolo.utils.datasets import pad 
#import torchvision.transforms as transforms
from yolo.utils.datasets import pad_to_square, resize, pad_to_square2

class UnalignedDataset(BaseDataset): # I/O for hybrid YOLOv3 + CycleGAN! Unsupported for batch data for YOLOv3
    def initialize(self, opt, normalized_labels = True):
        self.opt = opt
        self.root = opt.dataroot
        self.normalized_labels = normalized_labels 
        # self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        # self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.dir_A = os.path.join(opt.dataroot, 'A_train.txt') # A.txt contains a list of path/to/img1.jpg
        self.dir_B = os.path.join(opt.dataroot, 'B_train.txt')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt) # transform for cyclegan

        # prepare targets for yolo
        self.A_label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.A_paths
            ]
        # self.A_label_files = [
        #     path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt").replace("rainy/", "").replace("cloudy1000/", "").replace("sunny/", "").replace("night_or_night_and_rainy/", "")
        #     for path in self.A_paths
        #     ]
        self.B_label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt").replace("rainy/", "").replace("cloudy1000/", "").replace("sunny/", "").replace("night_or_night_and_rainy/", "")
            for path in self.B_paths
            ]


       
    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]
        A_path = A_path.strip('\n')
        B_path = B_path.strip('\n')

        #print('A_path = ', A_path)
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')


        #img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        tmp_A = transforms.ToTensor()(A_img)
        #print('\n**************************************************A_img.shape = ', tmp_A.shape)
        _, h, w = tmp_A.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        tmp_A, pad = pad_to_square2(tmp_A, 0)
        _, padded_h, padded_w = tmp_A.shape


        tmp_B = transforms.ToTensor()(B_img)
        #print('\n**************************************************A_img.shape = ', tmp_A.shape)
        _, hB, wB = tmp_B.shape
        h_factorB, w_factorB = (hB, wB) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        tmp_B, padB = pad_to_square2(tmp_B, 0)
        _, padded_hB, padded_wB = tmp_B.shape

        A_img = self.transform(A_img)
        B_img = self.transform(B_img)


        # ---------
        #  Label
        # ---------

        
        def label_path2bboxes(label_path, pad, h_factor, w_factor, padded_h, padded_w):
            tmp_targets = None
            if os.path.exists(label_path):
                boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
                # Extract coordinates for unpadded + unscaled image
                x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
                y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
                x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
                y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
                # Adjust for added padding
                x1 += pad[0]
                y1 += pad[2]
                x2 += pad[1]
                y2 += pad[3]
                # Returns (x, y, w, h) in scale [0, 1]
                boxes[:, 1] = ((x1 + x2) / 2) / padded_w
                boxes[:, 2] = ((y1 + y2) / 2) / padded_h
                boxes[:, 3] *= w_factor / padded_w
                boxes[:, 4] *= h_factor / padded_h

                #print('\nboxes x y w h: ', boxes)
                tmp_targets = torch.zeros((len(boxes), 6))
                tmp_targets[:, 1:] = boxes
                return tmp_targets

        label_path = self.A_label_files[index % len(self.A_paths)].rstrip()
        A_targets = label_path2bboxes(label_path, pad, h_factor, w_factor, padded_h, padded_w)

        label_path_B = self.B_label_files[index % len(self.B_paths)].rstrip()
        
        B_targets = label_path2bboxes(label_path_B, padB, h_factorB, w_factorB, padded_hB, padded_wB)

            #print('targets = ', targets)

        #targets = generate_YOLO_targets(self.bbox) # A_path = A_annotation
        
        # return {'A': A_img, 'B': B_img,
        #         'A_paths': A_path, 'B_paths': B_path,
        #          'targets': targets}


        return {'A': A_img, 'B': B_img,
                'A_paths': A_path, 'B_paths': B_path,
                'A_targets': A_targets, 'B_targets': B_targets} # add B_bbox, A_bbox

    def collate_fn(self, batch):
        # input images will be resized to 416
        # this collate_fn to suport batchSize >= 2

        #print('collate fn: ', zip(*batch))
        tmp = list(batch)
        
        #print('tmp = ', len(tmp))
        target_As = [data['A_targets'] for data in tmp if data['A_targets'] is not None]
        #print('targets_As = ', target_As)
        for i, boxes in enumerate(target_As):
            boxes[:, 0] = i

        target_As = torch.cat(target_As, 0) # BUG
        #print('target_As: ', target_As.shape)
        #print('target_As cat = ', target_As)

        target_Bs = [data['B_targets'] for data in tmp if data['B_targets'] is not None]
        for i, boxes in enumerate(target_Bs):
            boxes[:, 0] = i

        #print('\ntarget_Bs: ', target_Bs)
        #target_Bs = torch.cat(target_Bs, 0) # BUG        

        As = torch.stack([data['A'] for data in tmp])
        Bs = torch.stack([data['B'] for data in tmp])
        path_As = [data['A_paths'] for data in tmp]
        #path_As = torch.cat(path_As, 0)
        path_Bs = [data['B_paths'] for data in tmp]
        #path_Bs = torch.cat(path_Bs, 0)


        # paths, imgs, targets = list(zip(*batch))
        # # Remove empty placeholder targets
        # targets = [boxes for boxes in targets if boxes is not None]
        # # Add sample index to targets
        # for i, boxes in enumerate(targets):
        #     boxes[:, 0] = i
        # targets = torch.cat(targets, 0)
        # # Selects new image size every tenth batch
        # if self.multiscale and self.batch_count % 10 == 0:
        #     self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # # Resize images to input shape
        # imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        # self.batch_count += 1
        return {'A': As, 'B': Bs,
                'A_paths': path_As, 'B_paths': path_Bs,
                'A_targets': target_As, 'B_targets': target_Bs}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'

