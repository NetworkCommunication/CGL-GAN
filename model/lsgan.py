import torch.nn as nn
import torch
class Generator(nn.Module):
    def __init__(self, ims):
        super(Generator, self).__init__()

        self.init_size = 32 // 4
        self.l1 = nn.Sequential(nn.Linear(100, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Reshape(nn.Module):
    def __init__(self, init_size):
        super(Reshape, self).__init__()
        self.init_size = init_size

    def forward(self, input):
        return input.view(input.shape[0], 128, self.init_size, self.init_size)

class MixGenerator(nn.Module):
    def __init__(self, ims, N):
        super(MixGenerator, self).__init__()

        self.init_size = 32 // 4
        self.model = nn.Sequential(
            nn.Sequential(nn.Linear(100, 128 * self.init_size ** 2)),
            Reshape(init_size=self.init_size),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
        )

        modules = nn.ModuleList()
        for _ in range(N):
            modules.append(nn.Sequential(
                nn.BatchNorm2d(64, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 1, 3, stride=1, padding=1),
                nn.Tanh(),
            ))
        self.paths = modules

    def forward(self, z):
        hidden = self.model(z)
        img = []
        for path in self.paths:
            out = path(hidden)
            img.append(out.view((out.shape[0], *self.img_shape)))
        img = torch.cat(img, dim=0)
        return img


class Discriminator(nn.Module):
    def __init__(self, ims):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = 32 // 2 ** 4
        self.adv_layer = nn.Linear(128 * ds_size ** 2, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity