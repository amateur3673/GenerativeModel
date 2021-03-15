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

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256,512,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(512),
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

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

image_size = 64
batch_size = 64
latent_dim = 100

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

dataset = datasets.ImageFolder('/content/celeba',transform=transform)
dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
gen = Generator()
disc = Discriminator()
gen.apply(weights_init_normal)
disc.apply(weights_init_normal)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gen.to(device)
disc.to(device)

optim_gen = torch.optim.Adam(gen.parameters(),lr=1e-4,betas=(0.5,0.999))
optim_disc = torch.optim.Adam(disc.parameters(),lr=1e-4,betas=(0.5,0.999))

criterion = nn.MSELoss()
gen.train()
disc.train()

for epoch in range(21,41):
    print('Epoch {}:'.format(epoch))
    for i,data in enumerate(dataloader,0):
        print('Step {}:'.format(i),end=' ')
        imgs,_=data
        # Feeding the real sample to the discriminator
        imgs=imgs.to(device)
        disc.zero_grad()
        optim_disc.zero_grad()
        real_labels = torch.ones((imgs.shape[0],1),dtype=torch.float32,device=device)
        real_outputs = disc(imgs)
        real_loss = criterion(real_outputs,real_labels)
        # Feeding the fake sample to the discriminator
        noise = torch.randn(imgs.shape[0],latent_dim,1,1).to(device)
        fake_imgs = gen(noise).detach()
        fake_labels = torch.zeros((imgs.shape[0],1),dtype=torch.float32,device=device)
        fake_outputs = disc(fake_imgs)
        fake_loss = criterion(fake_outputs,fake_labels)
        D_loss = real_loss+fake_loss
        print('D_loss: {}'.format(D_loss.item()),end=' ')
        D_loss.backward()
        optim_disc.step()

        # Training generator
        gen.zero_grad()
        optim_gen.zero_grad()
        noise = torch.randn(batch_size,latent_dim,1,1).to(device)
        fake_imgs = gen(noise)
        fake_labels = torch.ones((batch_size,1),device=device)
        fake_outputs = disc(fake_imgs)
        gen_loss = criterion(fake_outputs,fake_labels)
        print('G_loss: {}'.format(gen_loss.item()))
        gen_loss.backward()
        optim_gen.step()
    
    show_image(gen,epoch)

torch.save(gen.state_dict(),'generator.pth')