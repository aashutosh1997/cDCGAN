import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.utils.tensorboard as tb

from models import Generator, Discriminator, weights_init

batch_size = 64
latent_dims = 128
train_logger = tb.SummaryWriter('logs', flush_secs=1)
transform=Compose([Resize(64),
                    ToTensor(),
                   Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])           
dataset = CIFAR10('data/', download=True, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator(10).to(device)
discriminator = Discriminator(10).to(device)
loss = nn.BCELoss()
optimizerG = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.9, 0.999))
optimizerD = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.9, 0.999))
one = 1
zero = 0
test_vector = torch.randn([batch_size, latent_dims], device=device)
test_labels = 5*torch.ones(batch_size, dtype=torch.int64, device=device) # 5: dogs, 3:cats
global_step = 0
for epoch in range(128):
    for idx, (images,lbls) in enumerate(train_loader):
        images = images.to(device)
        lbls = lbls.to(device)
        discriminator.zero_grad()
        batch_size = images.shape[0]
        labels = torch.full((batch_size,), one, dtype=images.dtype, device=device)
        outD = discriminator(images, lbls)
        errD = loss(outD, labels)
        errD.backward()
        Dx = outD.mean().item()
        
        z = torch.randn([batch_size, latent_dims], device=device)
        gen = generator(z, lbls)
        labels.fill_(zero)
        outF = discriminator(gen.detach(), lbls)
        errF = loss(outF, labels)
        errF.backward()
        Dz1 = outF.mean().item()
        err = errD + errF
        optimizerD.step()
        
        generator.zero_grad()
        labels.fill_(one)
        outD2 = discriminator(gen, lbls)
        errG = loss(outD2, labels)
        errG.backward()
        Dz2 = outD2.mean().item()
        optimizerG.step()
        if idx==0:
            image = generator(test_vector, test_labels)
            vutils.save_image(image.detach(),
                    'output/fake_samples_epoch_%03d.png' % (epoch),
                    normalize=True)
        
        train_logger.add_scalar('discriminator_loss', errD.item(), global_step=global_step)
        train_logger.add_scalar('generator_loss', errG.item(), global_step=global_step)
        train_logger.add_scalar('Dx', Dx, global_step=global_step)
        train_logger.add_scalar('DGz', Dz1/Dz2, global_step=global_step)
        global_step += 1
    if epoch%8 == 0:
        print("Epoch:[%d/128]"%(epoch))