import torch
import torch.nn as nn
from torch import autograd
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import os

class Critic(nn.Module):
    def __init__(self):
        super(Critic,self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1)
        self.conv3 = nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1)
        self.leaky_relu = nn.LeakyReLU()

        self.linear = nn.Linear(4*4*256,1)
    
    def forward(self,inputs):
        x = self.conv1(inputs)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)

        # Reshape
        x = x.view(-1,4*4*256)
        outputs = self.linear(x)
        return outputs

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.linear = nn.Linear(100,4*4*256)
        self.batch_norm1 = nn.BatchNorm2d(256)
        self.trans_conv1 = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.trans_conv2 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.trans_conv3 = nn.ConvTranspose2d(64,3,kernel_size=2,stride=2)
        self.activation = nn.ReLU()
        self.output_fn = nn.Tanh()
    
    def forward(self,inputs):
        x = self.linear(inputs)
        x = x.view(-1,256,4,4)
        x = self.batch_norm1(x)
        x = self.activation(x)
        x = self.trans_conv1(x)
        x = self.batch_norm2(x)
        x = self.activation(x)
        x = self.trans_conv2(x)
        x = self.batch_norm3(x)
        x = self.activation(x)
        x = self.trans_conv3(x)
        outputs = self.output_fn(x)
        return outputs

def process_image(imgs):
    """Keep image in range [-1,1]

    Args:
        imgs (torch.Tensor): 
    """
    return 2*imgs-1

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

def show_image(gen,num_iter,save_locs='visualize'):
    if not os.path.exists(save_locs):
        os.makedirs(save_locs)
    noise = torch.randn(100,100).to(device)
    fake_imgs = gen(noise).detach().cpu().permute(0,2,3,1).numpy()
    fake_imgs = 0.5*(fake_imgs+1)
    plt.figure(figsize=(10,10))
    for i in range(100):
        plt.subplot(10,10,i+1)
        plt.imshow(fake_imgs[i])
        plt.axis('off')
    plt.savefig(os.path.join(save_locs,'im_{}.png'.format(num_iter)))
    plt.show()

gen = Generator()
critic = Critic()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gen.to(device)
critic.to(device)
lambd = 10
n_critics = 5
batch_size = 64
num_iters = 150001
latent_dim = 100

transform = transforms.Compose([transforms.ToTensor()])
optim_gen = torch.optim.Adam(gen.parameters(),lr=0.0001,betas=(0,0.9))
optim_critic = torch.optim.Adam(critic.parameters(),lr=0.0001,betas=(0.0,0.9))

train_gen = torchvision.datasets.CIFAR10(root='./data',train=True,transform=transform,download=True)

dataloader = torch.utils.data.DataLoader(train_gen,batch_size=batch_size,shuffle=True)

gen.train()
critic.train()
#data_loader = iter(dataloader)
for iterations in range(num_iters):
    print('Iteration {}'.format(iterations))
    # Training the critic
    i=0
    for real_imgs,_ in dataloader:
        i+=1
        if i==6:
          break
        optim_critic.zero_grad()
        # Feeding real image to the critic and compute loss
        #real_imgs,_ = next(data_loader)
        real_imgs = process_image(real_imgs)
        real_imgs = real_imgs.to(device)
        critic_real = critic(real_imgs)
        critic_real = critic_real.mean()
        # Feeding the fake image to the critic
        noise = torch.randn(real_imgs.shape[0],latent_dim).to(device)
        fake_imgs = gen(noise).detach() #Remove the gradients
        critic_fake = critic(fake_imgs)
        critic_fake = critic_fake.mean()
        # Computing gradient penalty
        gradient_penalty = compute_gradient_penalty(critic,real_imgs,fake_imgs,device)

        critic_cost = -critic_real+critic_fake+lambd*gradient_penalty

        critic_cost.backward()
        print('Critic cost {}'.format(critic_cost))
        optim_critic.step()
    
    # Training genrator
    optim_gen.zero_grad()
    noise = torch.randn(batch_size,latent_dim).to(device)
    fake_imgs = gen(noise)
    critic_fake = -critic(fake_imgs)
    critic_fake = critic_fake.mean()
    critic_fake.backward()
    optim_gen.step()

    if (iterations%1000==0):
        show_image(gen,iterations)
torch.save(gen.state_dict(),'generator.pth')
torch.save(critic.state_dict(),'critic.pth')