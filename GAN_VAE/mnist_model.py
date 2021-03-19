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
        x = torch.sigmoid(self.linear4(x))
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
        y_hat = torch.sigmoid(self.linear3(feats2))

        return y_hat,feats1,feats2

class VAEGanTraining:

    def __init__(self,latent_dim):

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.disc = Discriminator()
        self.device = 'cuda'
        self.init_optim()
        self.gan_loss = nn.BCELoss()
        self.perceptual_loss = nn.MSELoss()
    
    def reparameterize(self,mean,logvar):
        eps = torch.randn(mean.shape).to(mean.device)
        sigma = torch.exp(0.5*logvar)
        return mean+eps*sigma
    
    def reconstruction_loss(self,x,x_prime):
        binary_crossentropy_loss = F.mse_loss(x_prime,x,reduction='sum')
        return binary_crossentropy_loss

    def init_optim(self):

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.disc.to(self.device)
        self.optim_encoder = torch.optim.Adam(self.encoder.parameters(),lr=1e-3)
        self.optim_decoder = torch.optim.Adam(self.decoder.parameters(),lr=1e-3)
        self.optim_disc = torch.optim.Adam(self.disc.parameters(),lr=1e-3)
        self.optim_ae = torch.optim.Adam(list(self.encoder.parameters())+list(self.decoder.parameters()),lr=1e-3)
    
    def kl_loss(self,mean,logvar):
        return 0.5*torch.sum(mean**2+logvar.exp()-logvar-1)

    def step_encoder(self,x):
        # Computing a step update of a encoder
        x_ = torch.clone(x)
        self.optim_ae.zero_grad()
        # Compute the represent z in latent space
        mean, logvar = self.encoder(x_)
        z = self.reparameterize(mean,logvar)
        # Decode and feed to disc
        x_tilde = self.decoder(z)
        x_ = x_.view(-1,28*28)
        x_out, x_feats1, x_feats2 = self.disc(x_)
        x_tilde_out, x_tilde_feats1, x_tilde_feats2 = self.disc(x_tilde)
        l_kl = self.kl_loss(mean,logvar)
        l_perceptual = self.perceptual_loss(x_feats1,x_tilde_feats1)+self.perceptual_loss(x_feats2,x_tilde_feats2)
        l_recon = self.reconstruction_loss(x_,x_tilde)

        encoder_loss = l_kl+l_perceptual+l_recon
        encoder_loss.backward()
        self.optim_encoder.step()

        return l_kl,l_perceptual,l_recon
    
    def decoder_step(self,x):
        # Computing a step of updating the decoder
        x_ = torch.clone(x)
        self.optim_decoder.zero_grad()
        # Compute the represent z in latent space
        mean,logvar = self.encoder(x_)
        z = self.reparameterize(mean,logvar)
        # Decode and feed to disc
        x_tilde = self.decoder(z)
        x_ = x_.view(-1,28*28)
        x_out, x_feats1, x_feats2 = self.disc(x_)
        x_tilde_out, x_tilde_feats1, x_tilde_feats2 = self.disc(x_tilde)
        z_p = torch.randn(z.shape).to(z.device)
        x_p = self.decoder(z_p)
        xp_out,_,_ = self.disc(x_p)

        real = torch.ones_like(x_tilde_out)
        l_perceptual = self.perceptual_loss(x_feats1,x_tilde_feats1)+ self.perceptual_loss(x_feats2,x_tilde_feats2)
        l_recon = self.reconstruction_loss(x_,x_tilde)
        l_gen = self.gan_loss(x_tilde_out,real)+self.gan_loss(xp_out,real)

        decoder_loss = l_perceptual+l_recon+0.2*l_gen
        decoder_loss.backward()
        self.optim_decoder.step()

        return decoder_loss

    def disc_step(self,x):

        x_ = torch.clone(x)
        self.optim_disc.zero_grad()
        mean, logvar = self.encoder(x_)
        z = self.reparameterize(mean,logvar)
        x_tilde = self.decoder(z)
        x_ = x_.view(-1,28*28)
        x_out, x_feats1, x_feats2 = self.disc(x_)
        x_tilde_out, x_tilde_feats1, x_tilde_feats2 = self.disc(x_tilde)

        # GAN loss
        x_out = x_out.view(x_out.shape[0])
        x_tilde_out = x_tilde_out.view(x_tilde_out.shape[0])
        real = torch.ones_like(x_out)
        fake = torch.zeros_like(x_tilde_out)

        z_p = torch.randn(z.shape).to(z.device)
        x_p = self.decoder(z_p)
        x_p_out,_,_ = self.disc(x_p)
        x_p_out = x_p_out.view(x_p_out.shape[0])
        loss_disc = self.gan_loss(x_tilde_out,fake)+self.gan_loss(x_out,real)+self.gan_loss(x_p_out,fake)

        percent_pred_real = np.mean(x_out.cpu().detach().numpy()>0.5)
        percent_pred_fake = np.mean(x_tilde_out.cpu().detach().numpy()<=0.5)

        
        loss_disc.backward()
        self.optim_disc.step()
        return loss_disc,percent_pred_real,percent_pred_fake
    
    def get_disc_loss(self,x):
        x_ = torch.clone(x)
        mean, logvar = self.encoder(x_)
        z = self.reparameterize(mean,logvar)
        x_tilde = self.decoder(z)
        x_ = x_.view(-1,28*28)
        x_out, x_feats1, x_feats2 = self.disc(x_)
        x_tilde_out, x_tilde_feats1, x_tilde_feats2 = self.disc(x_tilde)

        x_out = x_out.view(x_out.shape[0])
        x_tilde_out = x_tilde_out.view(x_tilde_out.shape[0])
        z_p = torch.randn(z.shape).to(z.device)
        x_p = self.decoder(z_p)
        x_p_out,_,_ = self.disc(x_p)
        x_p_out = x_p_out.view(x_p_out.shape[0])

        real = torch.ones_like(x_out)
        fake = torch.zeros_like(x_tilde_out)
        loss_disc = self.gan_loss(x_out,real) + self.gan_loss(x_tilde_out,fake) + self.gan_loss(x_p_out,fake)

        return loss_disc
    
    def one_step_training(self,x,train_disc=False):
        
        if train_disc:
            if self.get_disc_loss(x) > 0.4:
                loss_disc, percent_pred_real, percent_pred_fake = self.disc_step(x)
                return loss_disc,percent_pred_real,percent_pred_fake,None
            else:
                l_kl,l_perceptual,l_recon = self.step_encoder(x)
                self.decoder_step(x)
                return l_kl,l_perceptual,l_recon
        else:
            l_kl,l_perceptual,l_recon=self.step_encoder(x)
            self.decoder_step(x)
            return l_kl,l_perceptual,l_recon

def show_image(imgs,epoch):
    if not os.path.exists('visualize'):
        os.makedirs('visualize')
    imgs = imgs.view(imgs.shape[0],1,28,28)
    imgs = imgs.detach().cpu().numpy()
    plt.figure(figsize=(10,10))
    for i in range(100):
        plt.subplot(10,10,i+1)
        plt.imshow(imgs[i,0],cmap='gray')
        plt.axis('off')
    plt.savefig('visualize/im_{}.png'.format(epoch))
    plt.show()

if __name__=='__main__':
    dataset = MNIST('./data',train=True,transform=transforms.ToTensor())
    dataloader = DataLoader(dataset,batch_size=128,shuffle=True)
    vae_gan = VAEGanTraining(2)
    fix_latent = torch.randn(100,2).to('cuda')
    disc_wait = 8
    epochs = 100
    for epoch in range(epochs):
        for i,(x,_) in enumerate(dataloader):
            x = x.to('cuda')
            if i%disc_wait == 0:
                res = vae_gan.one_step_training(x,train_disc=True)
                if len(res)==4:
                    l_disc, percent_pred_real, percent_pred_fake,_ = res
                else:
                    l_kl, l_perceptual, l_recon = res
            else:
                l_kl, l_perceptual, l_recon = vae_gan.one_step_training(x,train_disc=False)
        
        print('Epoch [{}/{}]: l_recon: {:4f} l_kl: {:.4f} l_disc: {:.4f} l_percept: {:.4f} percent_pred_real: {:.4f} percent_pred_fake: {:.4f}'.format(
            epoch+1,epochs,l_recon,l_kl.data,l_disc.data,l_perceptual.data,percent_pred_real,percent_pred_fake
        ))
        outputs = vae_gan.decoder(fix_latent)
        show_image(outputs,epoch)
    
    torch.save(vae_gan.decoder.state_dict(),'decoder.pth')
    torch.save(vae_gan.encoder.state_dict(),'encoder.pth')