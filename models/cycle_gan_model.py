import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys

from yolo.utils.datasets import pad_to_square, resize 
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn.functional as F


def mse_loss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()


def yolo_data_transform(img, img_size=416, normalized_labels=True):
        # ---------
        #  Image
        # ---------

        # img_path = self.img_files[index % len(self.img_files)].rstrip()

        # # Extract image as PyTorch tensor
        # img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        # if len(img.shape) != 3:
        #     img = img.unsqueeze(0)
        #     img = img.expand((3, img.shape[1:]))
        #img = transforms.ToTensor()(img)
        #print('img.shape = ',img.shape)
        _, _, h, w = img.shape
        h_factor, w_factor = (h, w) if normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        
        return resize(img, img_size) 


class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids)


        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
            self.netFeat = networks.define_feature_network(opt.which_model_feat, self.gpu_ids)

            # load pretrained YOLO3
            self.YOLO3 = networks.define_pretrained_yolo3(opt.yolo_weight, opt.model_def, self.gpu_ids)


        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionFeat = mse_loss
            # yolo data transform
            self.yolo_data_transform = yolo_data_transform

            # utils
            #self.crop_img_by_bbox = crop_img_by_bbox
            
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.netG_B)
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
        print('-----------------------------------------------')

    def set_input(self, input):
        '''
        see unaligned_dataset.py

        return {'A': A_img, 'B': B_img,
                'A_paths': A_path, 'B_paths': B_path,
                'A_targets': A_targets}
        '''
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B'] # see unaligned_dataset.py
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        #print('\nA path: ', self.image_paths)
        self.A_targets = input['A_targets'] # prepared yolo taragets of A (synthesis traffic cones)
        self.B_targets = input['B_targets'] # get bboxes of B_img to crop traffic cones
        #self.targets = input['target_batch'] # see unaligned_dataset.py & dataloader YOLOv3

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG_A.forward(self.real_A) # use it
        self.rec_A = self.netG_B.forward(self.fake_B)

        self.real_B = Variable(self.input_B, volatile=True)
        self.fake_A = self.netG_B.forward(self.real_B)
        self.rec_B = self.netG_A.forward(self.fake_A)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD.forward(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD.forward(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        lambda_feat_AfB = self.opt.lambda_feat_AfB    
        lambda_feat_BfA = self.opt.lambda_feat_BfA

        lambda_feat_fArecB = self.opt.lambda_feat_fArecB
        lambda_feat_fBrecA = self.opt.lambda_feat_fBrecA

        lambda_feat_ArecA = self.opt.lambda_feat_ArecA
        lambda_feat_BrecB = self.opt.lambda_feat_BrecB

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A.forward(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B.forward(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss
        # D_A(G_A(A))
        self.fake_B = self.netG_A.forward(self.real_A) # fake_B
        pred_fake = self.netD_A.forward(self.fake_B)
        self.loss_G_A = self.criterionGAN(pred_fake, True)
        # D_B(G_B(B))
        self.fake_A = self.netG_B.forward(self.real_B)
        pred_fake = self.netD_B.forward(self.fake_A)
        self.loss_G_B = self.criterionGAN(pred_fake, True)
        # Forward cycle loss
        self.rec_A = self.netG_B.forward(self.fake_B) 
        
        # gamma = 1.
        # l_rec_A =  .2126 * self.rec_A[:,0]**gamma + .7152 * self.rec_A[:,1]**gamma + .0722 * self.rec_A[:,2]**gamma
        # l_real_A =  .2126 * self.real_A[:,0]**gamma + .7152 * self.real_A[:,1]**gamma + .0722 * self.real_A[:,2]**gamma
        
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        
        # Backward cycle loss
        self.rec_B = self.netG_A.forward(self.fake_A)

        # gamma = 1.
        # l_rec_B =  .2126 * self.rec_B[:,0]**gamma + .7152 * self.rec_B[:,1]**gamma + .0722 * self.rec_B[:,2]**gamma
        # l_real_B =  .2126 * self.real_B[:,0]**gamma + .7152 * self.real_B[:,1]**gamma + .0722 * self.real_B[:,2]**gamma
        
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B


        # print ('self.netFeat(self.real_A).parameters()', self.netFeat(self.real_A).parameters())
        # print ('self.netFeat(self.fake_B).parameters()', self.netFeat(self.fake_B).parameters())
        # print ('self.criterionFeat(self.netFeat(self.real_A), self.netFeat(self.fake_B)).parameters()', self.criterionFeat(self.netFeat(self.real_A), self.netFeat(self.fake_B)).parameters())

        # Perceptual Loss on Image Level
        self.feat_loss_AfB = self.criterionFeat(self.netFeat(self.real_A), self.netFeat(self.fake_B)) * lambda_feat_AfB    
        self.feat_loss_BfA = self.criterionFeat(self.netFeat(self.real_B), self.netFeat(self.fake_A)) * lambda_feat_BfA

        self.feat_loss_fArecB = self.criterionFeat(self.netFeat(self.fake_A), self.netFeat(self.rec_B)) * lambda_feat_fArecB
        self.feat_loss_fBrecA = self.criterionFeat(self.netFeat(self.fake_B), self.netFeat(self.rec_A)) * lambda_feat_fBrecA

        self.feat_loss_ArecA = self.criterionFeat(self.netFeat(self.real_A), self.netFeat(self.rec_A)) * lambda_feat_ArecA 
        self.feat_loss_BrecB = self.criterionFeat(self.netFeat(self.real_B), self.netFeat(self.rec_B)) * lambda_feat_BrecB 

        self.feat_loss = self.feat_loss_AfB + self.feat_loss_BfA + self.feat_loss_fArecB + self.feat_loss_fBrecA + self.feat_loss_ArecA + self.feat_loss_BrecB


        
        ###### yolov3 loss (regression + classification) #####
        # print('\nself.fake_B = ', self.fake_B.shape)
        self.fake_B_transformed = self.yolo_data_transform(self.fake_B)
        
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.fake_B_transformed = Variable(self.fake_B_transformed.type(Tensor), requires_grad=False)
        self.A_targets = Variable(self.A_targets.to(device), requires_grad=False)

        # print('\nfake_B_transformed = ', self.fake_B_transformed.shape)
        # print('\nself.A_targets shape = ', self.A_targets[0].shape)

        # print('\nself.A_targets = ', self.A_targets[0])

        ###### Perceptual Loss on Object Level #####

        # Crop traffic cones from images using bboxes
        #print('self.fake_B_transformed: ', self.fake_B_transformed.shape)
        # self.list_cones_syn = self.crop_img_by_bbox(self.fake_B_transformed, self.A_targets)
        # self.list_cones_real = self.crop_img_by_bbox(self.real_B, self.B_targets)

        # Extract traffic cones features 
  
        # self.feat_cone_syns = torch.stack([self.netFeat(cone_syn) for cone_syn in self.list_cones_syn], dim=0)
        # self.feat_cone_syn_1d = self.feat_cone_syns.mean(dim=0)

        # self.feat_cone_reals = torch.stack([self.netFeat(cone_real) for cone_real in self.list_cones_real], dim=0)
        # self.feat_cone_real_1d = self.feat_cone_reals.mean(dim=0)

        
        # Perceptual Loss at Object Level
        # self.feat_loss_object_level = self.criterionFeat(self.feat_cone_syn_1d, self.feat_cone_real_1d)


        with torch.no_grad():
            #self.yolo_loss, _ = self.YOLO3(self.fake_B_transformed, self.A_targets[0]) # create self.targets, this only worked for batchSize = 1
            self.yolo_loss, _ = self.YOLO3(self.fake_B_transformed, self.A_targets) # create self.targets

        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A \
                    + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B #\
                    # + self.feat_loss \
                    # + self.yolo_loss * 10 \
                    # + self.feat_loss_object_level * 20 

        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

    def get_current_errors(self):
        D_A = self.loss_D_A.data#[0]
        G_A = self.loss_G_A.data#[0]
        Cyc_A = self.loss_cycle_A.data#[0]
        D_B = self.loss_D_B.data#[0]
        G_B = self.loss_G_B.data#[0]
        Cyc_B = self.loss_cycle_B.data#[0]
        yolo_loss = self.yolo_loss.data 
        loss_G = self.loss_G.data
        if self.opt.identity > 0.0:
            idt_A = self.loss_idt_A.data#[0]
            idt_B = self.loss_idt_B.data#[0]
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A), ('idt_A', idt_A),
                                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B), ('idt_B', idt_B), ('yolo_loss', yolo_loss), ('loss_G', loss_G)])
        else:
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A),
                                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B), ('yolo_loss', yolo_loss), ('loss_G', loss_G)])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        rec_A = util.tensor2im(self.rec_A.data)
        real_B = util.tensor2im(self.real_B.data)
        fake_A = util.tensor2im(self.fake_A.data)
        rec_B = util.tensor2im(self.rec_B.data)
        if self.opt.identity > 0.0:
            idt_A = util.tensor2im(self.idt_A.data)
            idt_B = util.tensor2im(self.idt_B.data)
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A), ('idt_B', idt_B),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B), ('idt_A', idt_A)])
        else:
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D_B.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

