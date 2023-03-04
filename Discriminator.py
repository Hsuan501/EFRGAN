import torch
import torch.nn as nn
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, img_shape, num_classes):
        super(Discriminator, self).__init__()

        self.label_emb = nn.Sequential(
            nn.Embedding(num_classes, num_classes),
            nn.Linear(num_classes, int(np.prod(img_shape)))
            )

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)) * 2, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, label):
        d_input = torch.cat((img.view(img.size(0), -1), self.label_emb(label)), -1)
        validity = self.model(d_input)
        return validity

if __name__ == '__main__':
    from gan_utils.gan_dataset import DiscriminatorDataset
    from torch.utils.data import DataLoader
    d_dataset = DiscriminatorDataset('./dataset/emotion3Dface_label/Discriminator_label.txt')
    D_dataloader = DataLoader(d_dataset, batch_size=10, shuffle=True)
    D_iter = iter(D_dataloader)
    d_batch = next(D_iter)
    real_imgs = d_batch['img_tensor']
    real_labels = d_batch['label_tensor']
    d = Discriminator((1,224,224), 5)
    output = d(real_imgs, real_labels)
    print(output)



    """ useless code

    self.model = nn.Sequential(
            nn.Linear(num_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def fc_layer(self, input):
        input_size = input.size(dim=1)
        
        fc_layer = nn.Sequential(
        nn.Linear(input_size, 90),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(90, 30),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(30, 1),
        nn.Sigmoid(),
        )
        if torch.cuda.is_available():
            fc_layer.cuda()
        return fc_layer(input)
    """