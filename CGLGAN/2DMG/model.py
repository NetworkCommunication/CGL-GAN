import torch
import torch.nn as nn


# class Generator(nn.Module):
#     def __init__(self, img_shape, num_client):
#         super(Generator, self).__init__()
#         self.img_shape = img_shape
#         self.model = nn.Sequential(
#              nn.Linear(100, 128),
#              nn.LeakyReLU(0.2),
#              # nn.Linear(128, 128),
#              # nn.LeakyReLU(0.2),
#              nn.Linear(128, 2),
#              nn.Tanh()
#         )
#
#
#
#     def forward(self, z):
#         img = self.model(z)
#         img = img.view((img.shape[0], *self.img_shape))
#         return img


class Generator(nn.Module):
    def __init__(self, img_shape, num_client):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
             nn.Linear(100, 32),
             nn.LeakyReLU(0.2),
        )
        modules = nn.ModuleList()
        for _ in range(num_client):
            modules.append(nn.Sequential(
                # nn.Linear(256, 128),
                # nn.LeakyReLU(0.2),
                nn.Linear(32, 2),
                nn.Tanh()
            ))
        self.paths = modules

    def forward(self, z):
        img = []
        hidden_space = self.model(z)
        for path in self.paths:
            img.append(path(hidden_space))
        img = torch.cat(img, dim=0)
        return img

# ------------------------------------------
#                   G
class Discriminator(nn.Module):

    def __init__(self, ns=1):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),

        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity
