import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import numpy as np
from tqdm import tqdm
from loguru import logger
from datetime import datetime

from lib.config import cfg
from Generator import Generator
from Discriminator import Discriminator
from lib.gan_utils.lossfunc import compute_gradient_penalty
from lib.gan_utils import gan_dataset
from lib.gan_utils import util as gan_util

class Trainer(object):
    def __init__(self, G, D, config=None, device='cuda:0'):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.device = device
        self.batch_size = self.cfg.train.batch_size
        self.image_size = self.cfg.dataset.image_size
        self.global_step = 0
        self.log_flag = 0
        self.lambda_gp = self.cfg.train.lambda_gp
        self.n_critic = self.cfg.train.n_critic
        self.criterion = nn.BCEWithLogitsLoss()
        self.vaild = Variable(torch.full((self.batch_size, 1), 0.9).to(self.device), requires_grad=False)

        # load G and D
        self.generator = G.to(self.device)
        self.discriminator = D.to(self.device)
        self.generator.train()
        self.discriminator.train()
        for p in self.generator.resnet.parameters():
            p.requires_grad=False
        for p in self.generator.deca.parameters():
            p.requires_grad=False

        self.configure_optimizers()
        self.load_check_point()


        log_folder_path = os.path.join(self.cfg.output_dir, self.cfg.train.log_dir)
        os.makedirs(log_folder_path, exist_ok=True)
        logger.remove()
        logger.add(os.path.join(log_folder_path, 'train.log'))
        if self.cfg.train.write_summary:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=os.path.join(self.cfg.output_dir, self.cfg.train.log_dir))


    def configure_optimizers(self):
        self.opt_G = torch.optim.Adam(
                                    self.generator.parameters(),
                                    lr=self.cfg.G.lr,
                                    betas=(self.cfg.train.b1, self.cfg.train.b2))

        self.opt_D = torch.optim.Adam(
                                    self.discriminator.parameters(),
                                    lr=self.cfg.D.lr,
                                    betas=(self.cfg.train.b1, self.cfg.train.b2))

    def prepare_data(self):
        # dataset for G(netural img_parameters)
        self.g_dataset = gan_dataset.GeneratorDataset('/home/taros/Documents/Flame/new_ERFGAN/dataset/neutral')
        # dataset for D(happy img)
        
        self.d_dataset = gan_dataset.DiscriminatorDataset(str(self.cfg.D.label_path))

        self.G_dataloader = DataLoader(self.g_dataset, batch_size=self.batch_size, shuffle=True)
        self.G_iter = iter(self.G_dataloader)

        self.D_dataloader = DataLoader(self.d_dataset, batch_size=self.batch_size, shuffle=True)
        self.D_iter = iter(self.D_dataloader)

        logger.info(f'Generator data numbers: {len(self.g_dataset)}, Discriminator data numbers: {len(self.d_dataset)}')

    def load_check_point(self):

        if self.cfg.train.resume and os.path.exists(os.path.join(self.cfg.output_dir, 'G_model.tar')) and os.path.exists(os.path.join(self.cfg.output_dir, 'D_model.tar')):
            g_checkpoint = torch.load(os.path.join(self.cfg.output_dir, 'G_model.tar'))
            d_checkpoint = torch.load(os.path.join(self.cfg.output_dir, 'D_model.tar'))
            self.generator.load_state_dict(g_checkpoint['state_dict'])
            self.opt_G.load_state_dict(g_checkpoint['optimizer'])

            self.discriminator.load_state_dict(d_checkpoint['state_dict'])
            self.opt_D.load_state_dict(d_checkpoint['optimizer'])
            self.global_step = g_checkpoint['global_step']
            logger.info(f"resume training from {os.path.join(self.cfg.output_dir, 'model.tar')}")
            logger.info(f"training start from step {self.global_step}")
        # load model weights only
        else:
            logger.info('model path not found, start training from scratch')
            self.global_step = 0
            
        self.discriminator.train()
        self.generator.train()

    def D_training_step(self, fake_inputs, real_data):
        self.opt_D.zero_grad()
        fake_labels = Variable(torch.LongTensor(np.random.randint(1, cfg.num_classes, cfg.train.batch_size))).squeeze().to(self.device)
        fake_imgs = self.generator(fake_inputs,fake_labels).to(self.device)
    
        real_imgs = real_data['img_tensor'].to(self.device)
        real_labels = real_data['label_tensor'].to(self.device)

        #print(real_imgs.size(), real_labels.size(), fake_imgs.size(), fake_labels.size())

        # Real images
        real_validity = self.discriminator(real_imgs, real_labels)
        
        real_loss = self.criterion(real_validity, self.vaild)
        
        # Fake images
        fake_validity = self.discriminator(fake_imgs, fake_labels)
        fake_loss = self.criterion(fake_validity, Variable(torch.zeros(self.batch_size).unsqueeze(1).to(self.device), requires_grad=False))

        ''' loss func for wgan gap
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(self.discriminator, real_imgs.data, fake_imgs.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty
        '''

        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.opt_D.step()

        return d_loss

    def G_training_step(self, fake_inputs):
        self.opt_G.zero_grad()
        fake_labels = Variable(torch.LongTensor(np.random.randint(1, cfg.num_classes, cfg.train.batch_size))).squeeze().to(self.device)
 
        fake_imgs = self.generator(fake_inputs,fake_labels).to(self.device)
        # Loss measures generator's ability to fool the discriminator
        # Train on fake images
        validity = self.discriminator(fake_imgs, fake_labels)
        ''' loss func for wgan gap
        g_loss = -torch.mean(fake_validity)'''

        g_loss = self.criterion(validity, self.vaild)
        g_loss.backward()
        self.opt_G.step()
        return g_loss



    def fit(self): 
        self.prepare_data()

        iters_every_epoch = int(len(self.g_dataset)/self.batch_size)
        start_epoch = self.global_step//iters_every_epoch
        for epoch in range(start_epoch, self.cfg.train.max_epochs):
            for step in tqdm(range(iters_every_epoch), desc=f"Epoch[{epoch+1}/{self.cfg.train.max_epochs}]"):
                
                if epoch*iters_every_epoch + step < self.global_step:
                    continue
                try:
                    g_batch_data = next(self.G_iter)
                    d_batch_data = next(self.D_iter)
                except:
                    self.G_iter = iter(self.G_dataloader)
                    g_batch_data = next(self.G_iter)
                    
                    self.D_iter = iter(self.D_dataloader)
                    d_batch_data = next(self.D_iter)




                if step % self.n_critic == 0 or (step < 3 and epoch == 0):
                    d_loss = self.D_training_step(g_batch_data, d_batch_data)


                g_loss = self.G_training_step(g_batch_data)
    

                if step % self.n_critic == 0:
                    loss_info = f"[Epoch {epoch}/{self.cfg.train.max_epochs}] [Iter {step}/{iters_every_epoch}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}], Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
                    logger.info(loss_info)
                    if self.cfg.train.write_summary:
                        self.writer.add_scalar('g_loss', g_loss.item(), global_step=self.global_step)
                        self.writer.add_scalar('d_loss', d_loss.item(), global_step=self.global_step)

                
                if self.global_step % self.cfg.train.vis_steps == 0 and self.global_step != 0:
                    img_per_class, extra_img_num = divmod(cfg.train.batch_size, cfg.num_classes-1)
                    label_arr = np.array([i for i in range(1, cfg.num_classes)])
                    label_arr = np.repeat(label_arr, img_per_class)
                    
                    label_arr = np.append(label_arr, [cfg.num_classes-1 for i in range(extra_img_num)])
                    with torch.no_grad():
                        fake_labels = Variable(torch.LongTensor(label_arr)).squeeze().to(self.device)
                        fake_imgs = self.generator(g_batch_data,fake_labels).to(self.device)
                    
                    savepath = os.path.join(self.cfg.output_dir, self.cfg.train.vis_dir, f'{(epoch+1):03}_epoch-{step:04}.jpg')
                    fake_imgs = fake_imgs.repeat(1,3,1,1)
                    fake_imgs[:8] = g_batch_data['original_img'][:8].permute(0,3,1,2)
                    grid_image = gan_util.visualize_grid(fake_imgs, savepath, return_gird=True)
                    self.writer.add_image('train_images', (grid_image/255.).astype(np.float32).transpose(2,0,1), self.global_step)

                if self.global_step>0 and self.global_step % self.cfg.train.checkpoint_steps == 0:
                    generator_checkpoint = {
                        'global_step': self.global_step,
                        'state_dict': self.generator.state_dict(),
                        'optimizer': self.opt_G.state_dict(),
                        }
                    discriminator_checkpoint = {
                        'global_step': self.global_step,
                        'state_dict': self.discriminator.state_dict(),
                        'optimizer': self.opt_D.state_dict(),
                        }
                    torch.save(generator_checkpoint, os.path.join(self.cfg.output_dir, 'G_model' + '.tar'))
                    torch.save(discriminator_checkpoint, os.path.join(self.cfg.output_dir, 'D_model' + '.tar'))
                    if self.global_step % self.cfg.train.checkpoint_steps*10 == 0:
                            os.makedirs(os.path.join(self.cfg.output_dir, 'G_models'), exist_ok=True)
                            torch.save(generator_checkpoint, os.path.join(self.cfg.output_dir, 'G_models', f'{self.global_step:08}.tar'))
                            os.makedirs(os.path.join(self.cfg.output_dir, 'D_models'), exist_ok=True)
                            torch.save(discriminator_checkpoint, os.path.join(self.cfg.output_dir, 'D_models', f'{self.global_step:08}.tar'))

                self.global_step += 1
                




if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    
    generator_model = Generator()
    discriminator_model = Discriminator(cfg.D.img_shape, cfg.num_classes)
    trainer = Trainer(generator_model, discriminator_model)
    #trainer.load_check_point()
    trainer.fit()



'''
runtime test code:

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
end.record()
torch.cuda.synchronize()
print(start.elapsed_time(end))
'''
        


