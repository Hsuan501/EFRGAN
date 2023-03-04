import os, sys
from turtle import pos
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms

import numpy as np
from skimage.io import imread
from PIL import Image
import pylab as plt
from time import time



from lib.decalib.deca import DECA
from lib.decalib.utils import util
from lib.decalib.utils.config import cfg as deca_cfg
from lib.decalib.models.FLAME import FLAME

from lib.config import cfg
import time

torch.backends.cudnn.benchmark = True

''' 
1. input a 224x224 image(3D face)
2. use Resnet to extract features
3. flatten features
4. use 3 FC layer to predict parameter
'''

class FC_Layers(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_size)
            )
    
    def forward(self, x):
        x = self.fc_layer(x)
        return x

class Generator(nn.Module):
    def __init__(self, config=None, device='cuda'):
        super(Generator, self).__init__()
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.device = device
        # deca model that use for 3Dface construct
        
        self.class_num = cfg.num_classes
        self._setup_model(self.cfg.G)

    def _setup_model(self, model_cfg):
        exp_output_size = model_cfg.exp_outputsize

        resnet = models.resnet50()
        modules = list(resnet.children())[:-1]    # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.exp_predict_layer = FC_Layers(resnet.fc.in_features*2, exp_output_size)
        self.pose_predict_layer = FC_Layers(resnet.fc.in_features*2 , 3)
        #self.predict_layer = FC_Layers((resnet.fc.in_features+(self.class_num)*4), (exp_output_size+pose_output_size))
        
        self.label_emb = nn.Sequential(
            nn.Embedding(self.class_num, self.class_num),
            nn.Linear(self.class_num, resnet.fc.in_features)
            )
        


        #eval mode
        self.resnet.eval()
        self.exp_predict_layer.eval()
        self.pose_predict_layer.eval()

        
        self.flame = FLAME(model_cfg).to(self.device)
        self.deca = DECA(config = deca_cfg, device=self.device)
        self.resnet = self.resnet.to(self.device)
        self.exp_predict_layer = self.exp_predict_layer.to(self.device)
        self.pose_predict_layer = self.pose_predict_layer.to(self.device)
        self.label_emb = self.label_emb.to(self.device)


    def img_decoder(self, parameters, training_flag):

        output_tensor = None

        for i in range(parameters['exp'].size(dim=0)):
            sample_parm = {}
            for key, value in parameters.items():
                sample_parm[key] = value[i][:].unsqueeze(0)


            _ , visdict = self.deca.decode(sample_parm) #tensor
            image = util.tensor2image(visdict['shape_images'][0]).transpose(2,0,1)

            img_tensor = torch.from_numpy(image).float().unsqueeze(0)

            if training_flag:
                gray_transform= transforms.Grayscale(num_output_channels=1)
                img_tensor = gray_transform(img_tensor)

            if output_tensor is None:
                output_tensor = img_tensor
            else:
                output_tensor = torch.cat((output_tensor, img_tensor), dim=0)
        return output_tensor


        

    def forward(self, input_dict, label_tensor, demo_flag=False):
        # if train , the input will be a dict : [image tensor, deca_codedict]

        input_imgs = input_dict['img']
        input_parameters = input_dict['param']
        

        """for key, value in input_parameters.items():
            input_parameters[key] = torch.squeeze(value, 1).to(self.device)"""


        #input_batch = tensor_3D.unsqueeze(0) # create a mini-batch as expected by the model
        input_batch = input_imgs.to(self.device)

        #expression parameter predict(img features + embedded label)
        features = self.resnet(input_batch)
        

        features = features.reshape(features.size(0), -1)

        c = self.label_emb(label_tensor)

        
        if demo_flag == True and features.dim() > 1:
            features = features.squeeze()
        

        
        x = torch.cat([features, c], -1)

        exp_parameters = self.exp_predict_layer(x)
        pose_parameters = self.pose_predict_layer(x)

        if demo_flag == True and exp_parameters.dim() == 1:
            exp_parameters = exp_parameters.unsqueeze(0)
        
        if demo_flag == True and pose_parameters.dim() == 1:
            pose_parameters = pose_parameters.unsqueeze(0)

        exp_parameters *= 4
        exp_change_list = [0,1,4,6,8,9]
        for i in exp_change_list:
            exp_parameters[:, i] *= 2

        if self.cfg.emotion_type == 'sad':
            pose_parameters *= 0.1
        else:
            pose_parameters *= 0.135




        input_parameters['exp'] = exp_parameters
        input_parameters['pose'][:,3:] = pose_parameters
    
        output_tensor = self.img_decoder(input_parameters, True)

        return output_tensor
            


if __name__ == '__main__':
    from lib.gan_utils import gan_dataset
    from torch.autograd import Variable

    g_dataset = gan_dataset.GeneratorDataset('/home/taros/Documents/Flame/new_ERFGAN/dataset/neutral')
    G_dataloader = DataLoader(g_dataset, batch_size=3, shuffle=True)
    G_iter = iter(G_dataloader)
    g_batch_data = next(G_iter)

    input_parameters = g_batch_data['param']

    g= Generator()

    fake_labels = Variable(torch.LongTensor(np.random.randint(1, 2, 3))).squeeze().to('cuda')
 
    fake_imgs = g(g_batch_data, fake_labels)
    
    img = fake_imgs[0].numpy().transpose(1,2,0).astype('uint8')
    print(fake_imgs.size())

