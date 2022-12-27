import numpy as np
import torch
import torch.nn as nn
# MLP GAN
class Generator(nn.Module):
    def __init__(self, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(100, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view((img.shape[0], *self.img_shape))
        return img


class MixGenerator(nn.Module):
    def __init__(self, img_shape, num_client):
        super(MixGenerator, self).__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
             *block(100, 128, normalize=False),
             *block(128, 256),
             *block(256, 512),
        )

        modules = nn.ModuleList()
        for _ in range(num_client):
            modules.append(nn.Sequential(
                *block(512, 1024),
                nn.Linear(1024, int(np.prod(self.img_shape))),
                nn.Tanh()
            ))
        self.paths = modules

    def forward(self, z):
        img = []
        hidden = self.model(z)
        for path in self.paths:
            out = path(hidden)
            img.append(out.view((out.shape[0], *self.img_shape)))
        img = torch.cat(img, dim=0)
        return img


# ------------------------------------------
#                   G
class Discriminator(nn.Module):

    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 2),
            # nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity