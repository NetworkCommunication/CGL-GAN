import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(100, 32),
            nn.LeakyReLU(0.2),
            # nn.Linear(256, 128),
            # nn.LeakyReLU(0.2),
            nn.Linear(32, 2),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view((img.shape[0], *self.img_shape))
        return img


# ------------------------------------------
#                   G
class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity
