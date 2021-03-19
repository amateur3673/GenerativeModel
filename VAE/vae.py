import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import os

class Encoder(nn.Module):
    def __init__(self,latent_dim):
        super(Encoder,self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.linear1 = nn.Linear(4*4*256,latent_dim)
        self.linear2 = nn.Linear(4*4*256,latent_dim)
    
    def forward(self,inputs):
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.shape[0],-1)
        mean = self.linear1(x)
        logvar = self.linear2(x)
        return mean,logvar

class Decoder(nn.Module):

    def __init__(self,latent):
        super(Decoder,self).__init__()
        self.linear = nn.Linear(latent,4*4*256)
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(256,128,kernel_size=3,padding=1,stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(128,64,kernel_size=4,padding=1,stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(64,1,kernel_size=4,padding=1,stride=2),
            nn.Sigmoid()
        )
    
    def forward(self,inputs):
        x = self.linear(inputs)
        x = x.view(x.shape[0],256,4,4)
        x = self.block1(x)
        x = self.block2(x)
        outputs = self.block3(x)
        return outputs

class VAE(nn.Module):

    def __init__(self,latent_dim):
        super(VAE,self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    
    def reparameterize(self,mean,logvar,device='cuda'):
        # Reparameterization trick
        eps = torch.randn(mean.shape).to(device)
        std = torch.exp(0.5*logvar)
        return mean + eps*std
    
    def encode(self,x):
        return self.encoder(x)
    
    def decode(self,z):
        return self.decoder(z)

class VAELoss(nn.Module):

    def __init__(self,model):
        super(VAELoss,self).__init__()
        self.model = model
        self.bce_loss = nn.MSELoss(reduction='none')

    def forward(self,inputs):
        latent_mean,latent_logvar = self.model.encode(inputs)
        # Reparameter trick
        z = self.model.reparameterize(latent_mean,latent_logvar)
        m = z.shape[0]
        latent_var = torch.exp(latent_logvar)
        print(latent_mean.shape)
        KL_loss = 0.5*torch.mean(torch.sum(latent_mean**2+latent_var-latent_logvar-1,dim=1))

        x_logit = self.model.decode(z)
        recon_loss = 0.5*torch.mean(torch.sum(self.bce_loss(x_logit.view(x_logit.shape[0],-1),inputs.view(inputs.shape[0],-1)),dim=1))
        return KL_loss,recon_loss

def show_image(imgs,epoch):
    if not os.path.exists('visualize'):
        os.makedirs('visualize')
    imgs = imgs.detach().cpu().numpy()
    plt.figure(figsize=(10,10))
    for i in range(100):
        plt.subplot(10,10,i+1)
        plt.imshow(imgs[i,0])
        plt.axis('off')
    plt.savefig('visualize/im_{}.png'.format(epoch))
    plt.show()
if __name__=='__main__':
    dataset = MNIST('./data',train=True,transform=transforms.ToTensor())
    dataloader = DataLoader(dataset,batch_size=128,shuffle=True)
    model = VAE(16)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    losses = VAELoss(model)
    model.train()
    fix_vector = torch.randn(100,16).to('cuda')
    model.to('cuda')
    for epoch in range(30):
      print('Epoch {}'.format(epoch))
      for i,(imgs,_) in enumerate(dataloader):
        imgs = imgs.to('cuda')
        imgs = torch.where(imgs>=0.5,1.,0.)
        optimizer.zero_grad()
        KL_loss, recon_loss = losses(imgs)
        loss = KL_loss+recon_loss
        print('KL loss {}'.format(KL_loss.item()),end=' ')
        print('Recon loss {}'.format(recon_loss.item()))
        loss.backward()
        optimizer.step()
      outputs = model.decode(fix_vector)
      show_image(outputs,epoch) 
    torch.save(model.state_dict(),'vae.pth')