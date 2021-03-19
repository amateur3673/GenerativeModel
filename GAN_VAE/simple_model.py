import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import time
import os
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    def __init__(self,latent_dim):
        super(Encoder,self).__init__()
        self.linear1 = nn.Linear(784,512)
        self.linear2 = nn.Linear(512,128)
        self.linear3 = nn.Linear(128,64)
        self.mean_head = nn.Linear(64,latent_dim)
        self.logvar = nn.Linear(64,latent_dim)
    
    def forward(self,x):
        x = x.view(-1,28*28)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        mean = self.mean_head(x)
        logvar = self.logvar(x)
        return mean,logvar

class Decoder(nn.Module):
    def __init__(self,latent_dim):
        super(Decoder,self).__init__()
        self.linear1 = nn.Linear(latent_dim,64)
        self.linear2 = nn.Linear(64,128)
        self.linear3 = nn.Linear(128,512)
        self.linear4 = nn.Linear(512,784)
    
    def forward(self,z):
        x = F.relu(self.linear1(z))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.sigmoid(self.linear4(x))
        return x

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator,self).__init__()
        self.linear1 = nn.Linear(784,256)
        self.linear2 = nn.Linear(256,64)
        self.linear3 = nn.Linear(64,1)
    
    def forward(self,x):
        feats1 = F.relu(self.linear1(x))
        feats2 = F.relu(self.linear2(feats1))
        y_hat = F.sigmoid(self.linear3(feats2))

        return y_hat,feats1,feats2

def reconstruction_loss(x,x_prime):
    binary_crossentropy_loss = F.binary_cross_entropy(x_prime,x,reduction='sum')
    return binary_crossentropy_loss

def kl_loss(mean,logvar):
    return 0.5*torch.sum(mean**2+logvar.exp()-logvar-1)

def reparameterize(mean,logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn(mean.shape).to(mean.device)
    return mean+eps*std

def auto_encoder_step(data,ae_optim,enc,dec,disc,perceptual_loss):
    x,_ = data
    x = x.cuda()
    ae_optim.zero_grad()

    mean, logvar = enc(x) # compute the mean and variance
    z = reparameterize(mean,logvar)
    x_prime = dec(z) # reconstruct image

    x = x.view(-1,28*28)
    _,feats1_real,feats2_real = disc(x)
    _,feats1_fake,feats2_fake = disc(x_prime)

    l_recon = reconstruction_loss(x,x_prime) # reconstruction loss
    l_kl = kl_loss(mean,logvar) # KL loss
    l_perceptual = perceptual_loss(feats1_real,feats1_fake)+perceptual_loss(feats2_real,feats2_fake) # perceptual loss

    # Backward
    loss = l_kl+l_perceptual+l_recon
    loss.backward(retain_graph = True)
    ae_optim.step()

    return l_recon,l_kl,l_perceptual

def disc_step(data,disc_optim,enc,dec,disc,disc_loss):
    # Update the discriminator
    x,_ = data
    x = x.cuda()
    disc_optim.zero_grad()
    mean, logvar = enc(x)
    z = reparameterize(mean,logvar) # compute z from mean and logvar
    x_prime = dec(z) # reconstruct x

    x = x.view(-1,28*28)
    disc_real, feats1_real, feats2_real = disc(x)
    disc_fake, feats1_fake, feats2_fake = disc(x_prime)

    disc_real = disc_real.view(disc_real.shape[0])
    disc_fake = disc_fake.view(disc_fake.shape[0])

    real = torch.ones(disc_real.shape[0]).cuda()
    fake = torch.zeros(disc_fake.shape[0]).cuda()

    l_disc = disc_loss(disc_real,real)+disc_loss(disc_fake,fake)

    # Backward and step
    l_disc.backward()
    disc_optim.step()

    percent_pred_real = np.mean(disc_real.cpu().detach().numpy()>0.5)
    percent_pred_fake = np.mean(disc_fake.cpu().detach().numpy()<=0.5)

    return l_disc,percent_pred_real,percent_pred_fake

def get_disc_loss(data,enc,dec,disc,disc_loss):
    x,_ = data
    x = x.to('cuda')

    mean, logvar = enc(x)
    z = reparameterize(mean,logvar)
    x_prime = dec(z)

    x = x.view(-1,28*28)
    disc_real, feats1_real, feats2_real = disc(x)
    disc_fake, feats1_fake, feats2_fake = disc(x_prime)

    disc_real = disc_real.view(disc_real.shape[0])
    disc_fake = disc_fake.view(disc_fake.shape[0])

    real = torch.ones(disc_real.shape[0]).cuda()
    fake = torch.ones(disc_fake.shape[0]).cuda()

    l_disc = disc_loss(disc_real,real)+disc_loss(disc_fake,fake)
    return l_disc

def show_image(imgs,epoch):
    if not os.path.exists('visualize'):
        os.makedirs('visualize')
    imgs = imgs.view(imgs.shape[0],1,28,28)
    imgs = imgs.detach().cpu().numpy()
    plt.figure(figsize=(10,10))
    for i in range(100):
        plt.subplot(10,10,i+1)
        plt.imshow(imgs[i,0])
        plt.axis('off')
    plt.savefig('visualize/im_{}.png'.format(epoch))
    plt.show()

if __name__== '__main__':
    dataset = MNIST('./data',train=True,transform=transforms.ToTensor())
    dataloader = DataLoader(dataset,batch_size=128,shuffle=True)
    enc = Encoder(2)
    dec = Decoder(2)
    disc = Discriminator()
    enc = enc.to('cuda')
    dec = dec.to('cuda')
    disc = disc.to('cuda')
    disc_wait = 8
    epochs = 100
    disc_loss = nn.MSELoss()
    perceptual_loss = nn.MSELoss()

    ae_optim = torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),lr=1e-3)
    disc_optim = torch.optim.Adam(disc.parameters(),lr=1e-3)

    fix_latent = torch.randn(100,2).to('cuda')

    for epoch in range(epochs):
        start = time.time()
        for i,data in enumerate(dataloader):
            if i % disc_wait == 0: #update the discriminator
                if get_disc_loss(data,enc,dec,disc,disc_loss) > 0.4:
                    l_disc,percent_pred_real,percent_pred_fake = disc_step(data,disc_optim,enc,dec,disc,disc_loss)
                else:
                    l_recon, l_kl, l_perceptual = auto_encoder_step(data,ae_optim,enc,dec,disc,perceptual_loss)
            
            else:
                l_recon, l_kl, l_perceptual = auto_encoder_step(data,ae_optim,enc,dec,disc,perceptual_loss)
        
        elapse = time.time()-start
        print('Epoch [{}/{}]: l_recon: {:.4f} l_kl: {:.4f} l_disc: {:.4f} l_percept: {:.4f} time: {:.4f} real:{:.4f} fake: {:.4f}'.format(
            epoch+1,epochs,l_recon.data,l_kl.data,l_disc.data,l_perceptual.data,elapse,percent_pred_real,percent_pred_fake
        ))

        outputs = dec(fix_latent)
        show_image(outputs,epoch)
    
    torch.save(enc.state_dict(),'encoder.pth')