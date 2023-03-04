import os
import cv2
import argparse
import pylab as plt
from tqdm import tqdm
from os.path import join
from mpl_toolkits.axes_grid1 import ImageGrid


import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from lib.gan_utils import gan_dataset
from Generator import Generator

def plt_img(batch_data, img_list, index):

    img = img_list[index].numpy().transpose(1,2,0).astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_img = batch_data['original_img'][index]


    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                nrows_ncols=(1, 2),  # creates 2x2 grid of axes
                axes_pad=0.1,  # pad between axes in inch.
            )

    for ax, im in zip(grid, [original_img, img]):
        ax.imshow(im)
    
    plt.show()

def save_img(batch_data, img_list, save_folder, index=0):

    save_path = join(save_folder, batch_data['img_name'][index])
    img = img_list[index].numpy().transpose(1,2,0).astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    

    writeStatus = cv2.imwrite(save_path, img)
    if writeStatus is not True:
        print(f"Failed to save img to {save_path}.")



def main(args):
    os.makedirs(args.savefolder, exist_ok = True)


    save_folder = join(args.savefolder, args.label)
    os.makedirs(save_folder, exist_ok = True)

    checkpoint_path = join(args.checkpoint_folder, f'Gan_model_{args.label}/G_model.tar')

    g = Generator()

    if os.path.exists(checkpoint_path):
        print(f'trained model found. load {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        g.load_state_dict(checkpoint['state_dict'])
    else:
        print(f'please check model path: {checkpoint_path}')
        exit()


    g_dataset = gan_dataset.GeneratorDataset(args.inputfolder)
    data_length = len(g_dataset)

    if data_length < 10:

        G_dataloader = DataLoader(g_dataset, batch_size=data_length, shuffle=True)
        G_iter = iter(G_dataloader)
        g_batch_data = next(G_iter)

        fake_labels = Variable(torch.LongTensor([[1]]*data_length)).squeeze().to('cuda')
    
        fake_imgs = g(g_batch_data, fake_labels, demo_flag=True)

        save_img(g_batch_data, fake_imgs,  save_folder, index=0)

        #plt_img(g_batch_data, fake_imgs, index=0)

    else:
        batch_size = 10
        G_dataloader = DataLoader(g_dataset, batch_size=batch_size, shuffle=True)
        epochs = data_length // batch_size
        for epoch in tqdm(range(epochs)):
            G_iter = iter(G_dataloader)
            g_batch_data = next(G_iter)

            fake_labels = Variable(torch.LongTensor([[1]]*batch_size)).squeeze().to('cuda')
        
            fake_imgs = g(g_batch_data, fake_labels, demo_flag=True)

            #plt_img(g_batch_data, fake_imgs, index=19)

            for i in range(len(fake_imgs)):
                save_img(g_batch_data, fake_imgs,  save_folder, index=i)
            
            


            


    
    





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--inputfolder', default='./testSample/neutral', type=str,
                        help='path to the test data, it must be an image folder')

    parser.add_argument('-c', '--checkpoint_folder', default='model_checkpoint/', type=str,
                        help='path to the checkpoint directory')
                        
    parser.add_argument('-s', '--savefolder', default='./testSample/test_results', type=str,
                        help='path to the output directory, where results will be stored.')

    parser.add_argument('-l', '--label', default='happy', type=str,
                        help='use one of them: [happy, disgust, anger, sad]')
                                            
                        
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu')

    main(parser.parse_args())