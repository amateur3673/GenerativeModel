import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
import matplotlib.pyplot as plt
import os
hidden = Categorical(torch.ones(8)/8)

def generate_data(hidden,batch_size=256,num_modes=8,radius=1.,std=0.01):
    """
    Generate a random mixture model
    """
    theta = np.array([i*np.pi/4 for i in range(num_modes)])
    mean = np.array([radius*np.cos(theta),radius*np.sin(theta)]).T
    cov = [[std,0],[0,std]]
    mix_gaussian = [np.random.multivariate_normal(m,cov,batch_size) for m in mean]
    mix_gaussian = torch.tensor(mix_gaussian,dtype=torch.float32)
    h = hidden.sample(torch.Size([batch_size]))
    batch_idx = torch.arange(batch_size)

    mixture_gaussian = mix_gaussian[h,batch_idx]
    return mixture_gaussian

class Generator(nn.Module):

    def __init__(self,z_dim=256,hidden_dim=128,out_dim=2):
        super(Generator,self).__init__()
        self.linear_layer1 = nn.Linear(z_dim,hidden_dim)
        self.linear_layer2 = nn.Linear(hidden_dim,hidden_dim)
        self.linear_layer3 = nn.Linear(hidden_dim,out_dim)
        self.act = nn.Tanh()

    def forward(self,inputs):
        x = self.linear_layer1(inputs)
        x = self.act(x)
        x = self.linear_layer2(x)
        x = self.act(x)
        outputs = self.linear_layer3(x)
        return outputs

class Discriminator(nn.Module):

    def __init__(self,in_dim=2,hidden_dim=128):
        super(Discriminator,self).__init__()
        self.linear_layer1 = nn.Linear(in_dim,hidden_dim)
        self.linear_layer2 = nn.Linear(hidden_dim,hidden_dim)
        self.linear_layer3 = nn.Linear(hidden_dim,1)
        self.act = nn.Tanh()
    
    def forward(self,inputs):
        x = self.linear_layer1(inputs)
        x = self.act(x)
        x = self.linear_layer2(x)
        x = self.act(x)
        outputs = self.linear_layer3(x)
        return outputs.sigmoid()

def plot_samples(gen,fix_noise,num_iter):
    if not os.path.exists('visualize'):
        os.makedirs('visualize')
    outputs = gen(fix_noise)
    outputs = outputs.detach().cpu().numpy()
    plt.plot(outputs[...,0],outputs[...,1],'ro')
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.savefig('visualize/gen_{}.png'.format(num_iter))
    plt.show()

gen = Generator()
disc = Discriminator()
criterion = nn.BCELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gen.to(device)
disc.to(device)
gen_lr = 1e-3
disc_lr = 1e-4
n_iters = 40001
optim_G = torch.optim.Adam(gen.parameters(),lr=gen_lr,betas=(0.5,0.999))
optim_D = torch.optim.Adam(disc.parameters(),lr=disc_lr,betas=(0.5,0.999))
batch_size = 512
z_dim = 256
fix_noise = torch.randn(256,z_dim).to(device)
real_label = torch.ones(batch_size,dtype=torch.float32).to(device)
fake_label = torch.zeros(batch_size,dtype=torch.float32).to(device)

for i in range(n_iters):
    print('Step {}: '.format(i),end=' ')
    # Training discriminator
    disc.zero_grad()
    optim_D.zero_grad()
    # Feeding real samples and compute the loss
    real_samples = generate_data(hidden,batch_size)
    real_samples = real_samples.to(device)
    out_real = disc(real_samples)
    out_real = out_real.squeeze(-1)
    loss_real = criterion(out_real,real_label)
    # Feeding fake samples and compute loss
    noise = torch.randn(batch_size,z_dim).to(device)
    fake_samples = gen(noise).detach()
    out_fake = disc(fake_samples)
    out_fake = out_fake.squeeze(-1)
    loss_fake = criterion(out_fake,fake_label)
    loss_D = loss_real+loss_fake
    print('D_loss: {}'.format(loss_D.item()),end=' ')
    loss_D.backward()
    optim_D.step()
    # Training the generator
    gen.zero_grad()
    optim_G.zero_grad()
    noise = torch.randn(batch_size,z_dim).to(device)
    fake_samples = gen(noise)
    out_fake = disc(fake_samples)
    out_fake = out_fake.squeeze(-1)
    loss_G = criterion(out_fake,real_label)
    print('G loss {}'.format(loss_G.item()))
    loss_G.backward()
    optim_G.step()
    if (i%100==0):
        plot_samples(gen,fix_noise,i)

torch.save(gen.state_dict(),'gmm.pth')