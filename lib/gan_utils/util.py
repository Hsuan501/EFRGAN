import numpy as np
import torch
import torch.nn.functional as F
import cv2
import torchvision
from skimage.io import imread


def visualize_grid(output_tensor, savepath=None, size=224, dim=1, return_gird=True):
    '''
    image range should be [0,1]
    dim: 2 for horizontal. 1 for vertical
    '''
    assert dim == 1 or dim==2

    _,_,h,w = output_tensor.shape
    if dim == 2:
            new_h = size; new_w = int(w*size/h)
    elif dim == 1:
        new_h = int(h*size/w); new_w = size
    grid_tensor = torchvision.utils.make_grid(output_tensor.detach().cpu(), padding=1, nrow=4)
    grid_image = grid_tensor.numpy().transpose(1,2,0).astype('uint8')
    
    if savepath:
        cv2.imwrite(savepath, grid_image)
    if return_gird:
        return grid_image


if __name__ == '__main__':
    tensor = imread('/home/taros/Documents/Flame/EFRGan/model_checkpoint/Gan_model/train_images/001_epoch-0100.jpg')
    tensor = torch.from_numpy(tensor)

    gird = visualize_grid(tensor)