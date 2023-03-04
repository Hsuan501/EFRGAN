from cgi import test
import os, pickle, sys
import cv2
import random
import torch
import torchvision.transforms as transforms
import numpy as np
from skimage.io import imread
from glob import glob
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pylab as plt
from torch.autograd import Variable
from torch import LongTensor, device
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
from decalib.datasets import datasets as deca_dataset
from decalib.utils import util as deca_util
from pathlib import Path
import face_recognition


label_list = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness','surprise'] # class neutral is for generator, others for Discriminator

def image_preprocess(img_path):
    image = face_recognition.load_image_file(img_path, mode='RGB')
    face_locations = face_recognition.face_locations(image)
    top, right, bottom, left = face_locations[0]
    image = image[top:bottom, left:right]

    image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    if image.shape[2] != 3:
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

    return image



class GeneratorDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.deca = DECA(config = deca_cfg, device='cuda')

        if transform is not None:
            self.transform = transform
        else:
            self.transform= transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.path_list = []

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            self.path_list.append(file_path)
        

    def __len__(self):
        return len(self.path_list)
    
    def __getitem__(self, idx):
        try:
            img_path = self.path_list[idx]
        
            img = image_preprocess(img_path)
            if self.transform is not None:
                img_tensor = self.transform(img).to('cuda')

            with torch.no_grad():
                codedict = self.deca.encode(img_tensor[None,...])
            
            for key, value in codedict.items():
                codedict[key] = value.squeeze()
                
            sample_data = {'img': img_tensor, 'param': codedict, 'original_img': img, 'img_name': os.path.basename(img_path)}
            return sample_data
        except:
            print(img_path)
            

        




class DiscriminatorDataset(Dataset):
    def __init__(self, labelFile_path, transform=None):
        self.deca = DECA(config = deca_cfg, device='cuda')
        if transform is not None:
            self.transform = transform
        else:
            self.transform= transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
        label_file = open(labelFile_path, 'r')
        sample_datas = label_file.read().split('\n')
        path_list = []
        label_list = []
        for sample_data in sample_datas[:-1]:
            sample_dir, sample_label = sample_data.split(',')
            assert os.path.isfile(sample_dir), 'file not exist!'
            path_list.append(sample_dir)
            label_list.append(sample_label)

        self.path_list = path_list
        self.label_list = label_list
        
    def __len__(self):
        return len(self.label_list)
    
    def __getitem__(self, idx):
        img_path = self.path_list[idx]

        img = image_preprocess(img_path)
        if self.transform is not None:
            img_tensor = self.transform(img).to('cuda')

        with torch.no_grad():
            codedict = self.deca.encode(img_tensor[None,...])
            _ , visdict = self.deca.decode(codedict) #tensor

        gray_transform = transforms.Grayscale(num_output_channels=1)

        img_tensor = gray_transform(visdict['shape_images'][0])

        sample_label = self.label_list[idx]
        label_tensor = Variable(LongTensor([int(sample_label)])).squeeze()

        sample_data = {'img_tensor': img_tensor,  'label_tensor': label_tensor}
        return sample_data


if __name__ == '__main__':
    
    test_data = DiscriminatorDataset('/home/taros/Documents/Flame/new_ERFGAN/dataset/sad_label.txt')
    a = test_data[0]
    print(a['img_tensor'].size())

    #img = a['img_tensor'].cpu().numpy().transpose(1,2,0).astype('uint8')

    image = deca_util.tensor2image(a['img_tensor']).transpose(2,0,1)
    img = image.transpose(1,2,0).astype('uint8')

    
    plt.imshow(img)
    plt.show()

    """test_data = GeneratorDataset('/home/taros/Documents/Flame/new_ERFGAN/dataset/neutral')
    a = test_data[0]
    print(a['params'])
"""

