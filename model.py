import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils import read_images, init_params


#TODO Try batch norm in generator as in pytorch example

class Discriminator(nn.Module):

    def __init__(self, input_dim, input_ch, init_params=init_params):
        super().__init__()
        self.input_dim = input_dim
        self.input_ch = input_ch
        self.kernel = 5
        self.stride = 2
        self.padding = self.kernel // self.stride

        self.conv0_out_ch = 64
        self.conv1_out_ch = 128
        self.conv2_out_ch = 256
        self.conv3_out_ch = 512
        self.conv3_out_dim = int(np.ceil(self.input_dim / self.stride ** 4))

        self.conv_block = nn.Sequential(
            self.conv_layer(self.input_ch, self.conv0_out_ch),
            self.conv_layer(self.conv0_out_ch, self.conv1_out_ch, batch_norm=True),
            self.conv_layer(self.conv1_out_ch, self.conv2_out_ch, batch_norm=True),
            self.conv_layer(self.conv2_out_ch, self.conv3_out_ch, batch_norm=True)
        )

        self.linear = nn.Linear(self.conv3_out_ch * self.conv3_out_dim ** 2, 1)

        # Initialize parameter values
        self.apply(init_params)

    def conv_layer(self, in_ch, out_ch, batch_norm=False):
        ''' Standard convolutional layer '''
        # bias=False https://stackoverflow.com/questions/46256747
        sequence = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, self.kernel, self.stride, self.padding, bias=False),
            nn.LeakyReLU(0.2)
        )
        if batch_norm:
            sequence.add_module('batch_norm', nn.BatchNorm2d(out_ch))
        return sequence

    def forward(self, x):
        ''' Discriminator forward pass '''
        x = self.conv_block(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return x.squeeze()


class Generator(nn.Module):

    def __init__(self, input_size, output_dim, output_ch, init_params=init_params):
        super().__init__()
        self.input_size = input_size
        self.output_dim = output_dim
        self.output_ch = output_ch
        self.kernel = 5
        self.stride = 2
        self.padding = self.kernel // self.stride

        self.deconv0_in_dim = int(np.ceil(self.output_dim / self.stride ** 4))
        self.deconv0_in_ch = 512
        self.deconv0_out_ch = 256
        self.deconv1_out_ch = 128
        self.deconv2_out_ch = 64
        
        self.linear = nn.Linear(self.input_size, self.deconv0_in_ch * self.deconv0_in_dim ** 2) 

        self.deconv_block = nn.Sequential(
            self.deconv_layer(self.deconv0_in_ch, self.deconv0_out_ch),
            self.deconv_layer(self.deconv0_out_ch, self.deconv1_out_ch),
            self.deconv_layer(self.deconv1_out_ch, self.deconv2_out_ch),
            self.deconv_layer(self.deconv2_out_ch, self.output_ch)
        )

        # Initialize parameter values
        self.apply(init_params)

    def deconv_layer(self, in_ch, out_ch):
        ''' Standard transposed convolutional layer '''
        sequence = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, self.kernel, self.stride, self.padding, bias=False),
            nn.ReLU()
        )
        return sequence

    def forward(self, x):
        ''' Generator forward pass '''
        x = self.linear(x)
        x = x.reshape(x.shape[0], self.deconv0_in_ch, self.deconv0_in_dim, self.deconv0_in_dim)
        x = self.deconv_block(x)
        return torch.tanh(x)

    
class DCGAN(object):

    def __init__(self, discriminator, generator):
        self.discriminator = discriminator
        self.generator = generator

    def optimizer(self, parameters):
        return torch.optim.Adam(params=parameters, lr=0.0002, betas=(0.9, 0.5))

    def train(self, image_dataset, n_epochs, batch_size=32):

        dis_optimizer = self.optimizer(self.discriminator.parameters())
        gen_optimizer = self.optimizer(self.generator.parameters())

        loss_function = nn.BCEWithLogitsLoss()
        
        for epoch in range(n_epochs):
            print('Epoch:', epoch)

            data_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True)

            for image_batch in data_loader:

                batch_size = image_batch.shape[0]
                
                # Generate images from random inputs
                # ganhacks suggests normal dist (uniform dist in original TF)
                z_batch = torch.randn(batch_size, generator.input_size)
                gen_image_batch = self.generator(z_batch)

                # Discriminator forward pass with real and fake batches
                logits_real = self.discriminator(image_batch).squeeze()
                logits_fake = self.discriminator(gen_image_batch).squeeze()

                dis_loss_real = loss_function(logits_real, torch.ones(batch_size))
                dis_loss_fake = loss_function(logits_fake, torch.zeros(batch_size))
                dis_loss = dis_loss_real + dis_loss_fake

                # Discriminator backwards pass and parameter update
                dis_optimizer.zero_grad()
                # N.B. we need to retain graph as we are not re-running gen_image_batch https://stackoverflow.com/questions/46774641
                dis_loss.backward(retain_graph=True)
                dis_optimizer.step()
                
                # Generator forward pass 
                # Target values explanation https://arxiv.org/pdf/1701.00160.pdf section:3.2.3
                logits_fake = self.discriminator(gen_image_batch)
                gen_loss = loss_function(logits_fake, torch.ones(batch_size))

                # Generator backwards pass and parameter update
                gen_optimizer.zero_grad()
                gen_loss.backward()
                gen_optimizer.step()


if __name__ == '__main__':

    image_ch = 3
    image_dim = 150
    z_size = 100

    image_dataset = read_images('images')

    discriminator = Discriminator(input_dim=image_dim, input_ch=image_ch)
    generator = Generator(input_size=z_size, output_dim=image_dim, output_ch=image_ch)

    dcgan = DCGAN(discriminator, generator)

    dcgan.train(image_dataset, 10)

    print('end')