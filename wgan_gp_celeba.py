import torch
import torch.nn as nn
from torch import autograd
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image

class Critic(nn.Module):
    def __init__(self):
        super(Critic,self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256,512,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2)
        )
        self.linear_layer = nn.Linear(4*4*512,1)
    
    def forward(self,inputs):
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(-1,4*4*512)
        outputs = self.linear_layer(x)
        return outputs

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(100,512,kernel_size=4,stride=1,padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(512,256,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv_out = nn.ConvTranspose2d(64,3,kernel_size=4,stride=2,padding=1)
        self.out_activation = nn.Tanh()
    
    def forward(self,inputs):
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.conv_out(x)
        outputs = self.out_activation(x)
        return outputs

def compute_gradient_penalty(critic,real_imgs,fake_imgs,device='cuda'):
    
    # Generating a random number
    eps = torch.rand(1,1,1,1)
    b,c,w,h = real_imgs.shape
    eps = eps.repeat(b,c,w,h).to(device)
    sample_imgs = eps*real_imgs+(1-eps)*fake_imgs
    sample_imgs.requires_grad = True
    outputs = critic(sample_imgs)
    # Computing gradients
    gradients = autograd.grad(outputs=outputs,inputs=sample_imgs,
                             grad_outputs=torch.ones(outputs.size(),device=device),retain_graph=True,create_graph=True,only_inputs=True)[0]
    gradients = gradients.view(gradients.shape[0],-1)
    gradient_penalty = torch.mean((torch.norm(gradients,dim=-1)-1)**2)
    return gradient_penalty

def show_image(gen,num_iter,save_locs='visualize',device='cuda'):
    
    if not os.path.exists(save_locs):
        os.makedirs(save_locs)
    noise = torch.randn(100,100,1,1).to(device)
    fake_imgs = gen(noise).detach().cpu().permute(0,2,3,1).numpy()
    fake_imgs = 0.5*(fake_imgs+1)
    plt.figure(figsize=(10,10))
    for i in range(100):
        plt.subplot(10,10,i+1)
        plt.imshow(fake_imgs[i])
        plt.axis('off')
    plt.savefig(os.path.join(save_locs,'im_{}.png'.format(num_iter)))
    plt.show()

image_size = 64
batch_size = 64
lambd = 10.
n_critics = 5
num_iters = 120001
latent_dim = 100

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

dataset = datasets.ImageFolder('/home/dell/data/celeba',transform=transform)
dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
gen = Generator()
critic = Critic()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gen.to(device)
critic.to(device)

optim_gen = torch.optim.Adam(gen.parameters(),lr=1e-4,betas=(0.0,0.9))
optim_critic = torch.optim.Adam(critic.parameters(),lr=1e-4,betas=(0.0,0.9))

gen.train()
critic.train()
for iteration in num_iters:
    print('Iteration {}:'.format(iteration))
    i = 0
    # Train the critic
    for real_imgs,_ in dataset:
        optim_critic.zero_grad()
        # Feeding the real image to the critic
        real_imgs = real_imgs.to(device)
        critic_real = critic(real_imgs)
        critic_real = critic_real.mean()
        # Feed the fake image to the critic
        noise = torch.randn(real_imgs.shape[0],latent_dim,1,1).to(device)
        fake_imgs = gen(noise).detach()
        critic_fake = critic(fake_imgs)
        critic_fake = critic_fake.mean()
        # Computing the gradient penalty
        gradient_penalty = compute_gradient_penalty(critic,real_imgs,fake_imgs,device=device)

        critic_cost = -critic_real+critic_fake+lambd*gradient_penalty
        critic_cost.backward()
        print('Critic cost {}'.format(critic_cost))
        optim_critic.step()
        i+=1
        if i==5: break
    
    # Train the generator
    optim_gen.zero_grad()
    noise = torch.randn(batch_size,latent_dim,1,1).to(device)
    fake_imgs = gen(noise)
    critic_fake = -critic(fake_imgs)
    critic_fake = critic_fake.mean()
    critic_fake.backward()
    optim_gen.step()

    if (iteration%1000==0):
        show_image(gen,iteration)
torch.save(critic.state_dict(),'critic.pth')
torch.save(gen.state_dict(),'generator.pth')