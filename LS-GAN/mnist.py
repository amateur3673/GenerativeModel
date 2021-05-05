import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from torch import autograd

class Generator(nn.Module):

    def __init__(self,z_dim=100,latent_dim=64):
        super(Generator,self).__init__()
        self.block1 = nn.Sequential(
            nn.Linear(z_dim,latent_dim*64),
            nn.BatchNorm1d(latent_dim*64),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim*4,latent_dim*2,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(latent_dim*2),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim*2,latent_dim,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU()
        )
        self.conv4 = nn.ConvTranspose2d(latent_dim,1,kernel_size=4,stride=2,padding=1)
    
    def forward(self,inputs):
        x = self.block1(inputs)
        x = x.view(x.shape[0],-1,4,4)
        x = self.block2(x)
        x = self.block3(x)
        outputs = torch.tanh(self.conv4(x))
        return outputs

class Discriminator(nn.Module):
    def __init__(self,latent_dim = 64):
        super(Discriminator,self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1,latent_dim,kernel_size=5,stride=2,padding=2),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(latent_dim,latent_dim*2,kernel_size=5,stride=2,padding=2),
            nn.BatchNorm2d(latent_dim*2),
            nn.ReLU(0.2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(latent_dim*2,latent_dim*4,kernel_size=5,stride=2,padding=2),
            nn.BatchNorm2d(latent_dim*4),
            nn.LeakyReLU(0.2)
        )
        self.linear = nn.Linear(latent_dim*4*4*4,1)
    
    def forward(self,inputs):
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.shape[0],-1)
        return self.linear(x)

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

def init_weight(m):
    if type(m) == nn.Conv2d or type(m)== nn.ConvTranspose2d or type(m)==nn.Linear:
        torch.nn.init.normal_(m.weight,0.,0.02)
        m.bias.data.fill_(0.)

def compute_distance(real,fake):
    real = real.view(real.shape[0],-1)
    fake = fake.view(fake.shape[0],-1)
    return torch.norm(real-fake,p=1,dim=1,keepdim=True)

def gradient_penalty(samples,model):
    """Compute the gradient of discriminator respect to the input

    Args:
        samples: input samples
        disc : discriminator
    """
    samples.requires_grad = True
    outputs = model(samples)
    gradients = autograd.grad(outputs=outputs,inputs=samples,
                             grad_outputs = torch.ones(outputs.size(),device=samples.device),retain_graph=True,create_graph=True,only_inputs=True)[0]
    gradients = gradients.view(gradients.shape[0],-1)
    gradients_penalty = torch.mean(torch.norm(gradients,p=2,dim=1))
    return gradients_penalty

def train_model(gen,disc,dataloader,lambd=1000,lr=1e-3,epochs=25,device='cuda',gamma=5.):
    
    optim_gen = torch.optim.Adam(gen.parameters(),lr=lr,betas=(0.5,0.999))
    optim_disc = torch.optim.Adam(disc.parameters(),lr=lr,betas=(0.5,0.999))
    fix_dim = torch.randn(100,100).to(device)
    gen.to(device)
    disc.to(device)
    for epoch in range(epochs):
        print('Epoch {}:'.format(epoch))
        for i, (X,y) in enumerate(dataloader):
            X = X.to(device)
            # Train the discriminator
            optim_disc.zero_grad()
            output_real = disc(X)
            z = torch.randn(X.shape[0],100).to(device)
            fake_imgs = gen(z).detach()
            output_fake = disc(fake_imgs)
            delta = compute_distance(X,fake_imgs)
            gradient = gradient_penalty(X,disc)
            disc_loss = torch.mean(output_real)+lambd*torch.mean(F.relu(delta+output_real-output_fake))
            disc_loss.backward()
            optim_disc.step()
            # Train the generator
            optim_gen.zero_grad()
            z = torch.randn(X.shape[0],100).to(device)
            fake_imgs = gen(z)
            output_fake = disc(fake_imgs)
            gen_loss = torch.mean(output_fake)
            gen_loss.backward()
            optim_gen.step()
            if i%25 == 0:
               print('Disc loss {}. Gen loss: {}'.format(disc_loss.data,gen_loss.data))
        vis_imgs = gen(fix_dim)
        show_image(vis_imgs,epoch)


if __name__=='__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])
    dataset = MNIST('./data',train=True,transform = transform)
    dataloader = DataLoader(dataset,batch_size=64,shuffle=False)
    gen = Generator()
    disc = Discriminator()
    train_model(gen,disc,dataloader)