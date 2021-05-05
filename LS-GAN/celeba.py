import torch.nn as nn
import torch
from torch import autograd
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.nn import functional as F
import os
import matplotlib.pyplot as plt

class GenBaseBlock(nn.Module):
    def __init__(self,input_dim,output_dim,kernel_size,stride,padding):
        super(GenBaseBlock,self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(input_dim,output_dim,kernel_size,stride,padding),
            nn.BatchNorm2d(output_dim),
            nn.ReLU()
        )
    
    def forward(self,inputs):
        return self.block(inputs)

class Generator(nn.Module):

    def __init__(self,zdim=100,base_dim=64,output_channel=3):
        super(Generator,self).__init__()
        self.block1 = GenBaseBlock(zdim,base_dim*8,kernel_size=4,stride=4,padding=0)
        self.block2 = GenBaseBlock(base_dim*8,base_dim*4,kernel_size=4,stride=2,padding=1)
        self.block3 = GenBaseBlock(base_dim*4,base_dim*2,kernel_size=4,stride=2,padding=1)
        self.block4 = GenBaseBlock(base_dim*2,base_dim,kernel_size=4,stride=2,padding=1)
        self.conv_transpose = nn.ConvTranspose2d(base_dim,output_channel,kernel_size=4,stride=2,padding=1)
    
    def forward(self,inputs):
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.conv_transpose(x)
        return torch.tanh(x)

class DisBaseBlock(nn.Module):

    def __init__(self,input_dim,output_dim,kernel_size,stride,padding):
        super(DisBaseBlock,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_dim,output_dim,kernel_size,stride,padding),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self,inputs):
        return self.block(inputs)

class Discriminator(nn.Module):
    def __init__(self,base_dim = 64,im_channel=3):
        super(Discriminator,self).__init__()
        self.block1 = DisBaseBlock(im_channel,base_dim,4,2,1)
        self.block2 = DisBaseBlock(base_dim,base_dim*2,4,2,1)
        self.block3 = DisBaseBlock(base_dim*2,base_dim*4,4,2,1)
        self.block4 = DisBaseBlock(base_dim*4,base_dim*8,4,2,1)
        self.conv = nn.Conv2d(base_dim*8,1,kernel_size=4)
    
    def forward(self,inputs):
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        outputs = self.conv(x)
        return outputs.view(outputs.shape[0],1)

def weight_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.normal_(m.weight,0.,0.02)
    elif type(m) == nn.BatchNorm2d:
        torch.nn.init.normal_(m.weight,1.0,0.02)
        m.bias.data.fill_(0.)

def compute_gradient_penalty(samples,model):

    samples.requires_grad = True
    outputs = model(samples)
    gradients = autograd.grad(outputs=outputs,inputs=samples,
                             grad_outputs=torch.ones(outputs.size(),device=samples.device),retain_graph=True,create_graph=True,only_inputs=True)[0]
    gradients = gradients.view(gradients.shape[0],-1)
    gradient_penalty = torch.mean(torch.norm(gradients,p=2,dim=1))
    return gradient_penalty

def compute_distance(X,Y,p=1):
    '''
    Compute the Lp distance between X and Y
    '''
    return torch.norm(X.view(X.shape[0],-1)-Y.view(Y.shape[0],-1),p=p,dim=1)

def show_image(vis_img,epoch):
    if not os.path.exists('visualize'):
        os.makedirs('visualize')
    img = vis_img.view(vis_img.shape[0],3,64,64)
    img = img.detach().cpu().permute(0,2,3,1)
    img = 0.5*(img+1)
    img = img.numpy()
    plt.figure(figsize=(15,15))
    for i in range(100):
        plt.subplot(10,10,i+1)
        plt.imshow(img[i])
        plt.axis('off')
    plt.savefig('visualize/im_{}.png'.format(epoch))
    plt.show()

def train_model(dataloader,z_dim=100,base_dim=64,lrD=2e-4,lrG=2e-4,epochs = 51, lambd=2e-4, gamma = 0.25):
    
    disc = Discriminator(base_dim,im_channel=3)
    gen = Generator(z_dim,base_dim,3)
    disc.apply(weight_init)
    gen.apply(weight_init)
    optim_D = torch.optim.Adam(disc.parameters(),lr=lrD,betas=(0.5,0.999))
    optim_G = torch.optim.Adam(gen.parameters(),lr=lrG,betas=(0.5,0.999))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    disc.to(device)
    gen.to(device)
    disc.train()
    gen.train()

    fix_noise = torch.randn(100,z_dim,1,1).to(device)

    for epoch in range(epochs):
        print('Epoch {}'.format(epoch))
        for i,(X,_) in enumerate(dataloader):
            #Train D
            optim_D.zero_grad()
            X = X.to(device)
            z = torch.randn(X.shape[0],z_dim,1,1).to(device)
            real_outputs = disc(X)
            fake_imgs = gen(z).detach()
            fake_outputs = disc(fake_imgs)
            distance = compute_distance(X,fake_imgs)*lambd
            gp = compute_gradient_penalty(X,disc)
            disc_loss = torch.mean(F.leaky_relu(real_outputs-fake_outputs+distance,0.))+gamma*gp
            disc_loss.backward()
            
            optim_D.step()
            # Train G
            gen.zero_grad()
            z = torch.randn(X.shape[0],z_dim,1,1).to(device)
            fake_imgs = gen(z)
            fake_outputs = disc(fake_imgs)
            G_loss = torch.mean(fake_outputs)
            G_loss.backward()
            optim_G.step()
            
            if i%20==0:
                print('Step: {}. Disc loss: {}. Gen loss: {}'.format(i,disc_loss.data,G_loss.data))
            
        vis_img = gen(fix_noise)
        show_image(vis_img,epoch)



if __name__=='__main__':
    image_size = 64
    batch_size = 64
    zdim = 100
    lrD = 2e-4
    lrG = 2e-4
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    dataset = datasets.ImageFolder('/content/celeba',transform=transform)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

    train_model(dataloader)