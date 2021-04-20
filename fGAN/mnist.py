import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from torch.nn import functional as F

class Generator(nn.Module):

    def __init__(self):
        super(Generator,self).__init__()
        self.block1 = nn.Sequential(
            nn.Linear(100,1200),
            nn.BatchNorm1d(1200),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Linear(1200,1200),
            nn.BatchNorm1d(1200),
            nn.ReLU()
        )
        self.linear3 = nn.Linear(1200,784)
    
    def forward(self,inputs):
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.linear3(x)
        return torch.sigmoid(x)

class T_function(nn.Module):

    def __init__(self):
        super(T_function,self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(784,1200),
            nn.ELU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(1200,1200),
            nn.ELU()
        )
        self.linear3 = nn.Linear(1200,1)
    
    def forward(self,inputs):
        x = inputs.view(inputs.shape[0],-1)
        x = F.dropout(self.linear1(x),0.3)
        x = F.dropout(self.linear2(x),0.3)
        x = F.dropout(self.linear3(x),0.3)
        return x

def sampling(batch_size,in_dims,device='cuda'):
    noise = torch.randn(batch_size,in_dims)
    return noise.to(device)

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
    batch_size = 128
    from torchvision.datasets import MNIST
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from simple_estimate import Estimate_function
    mnist = MNIST('./data',train=True,transform=transforms.ToTensor())
    dataloader = DataLoader(mnist,batch_size=128,shuffle=False)
    function = Estimate_function('pearson')
    num_epochs = 70
    
    device = 'cuda'
    fix_noise = sampling(100,100,device=device)
    gen = Generator()
    gen.to(device)
    gen.train()
    T = T_function()
    T.to(device)
    T.train()
    optim_gen = torch.optim.Adam(gen.parameters(),lr = 2e-4, betas=(0.5,0.999))
    optim_T = torch.optim.Adam(T.parameters(),lr = 2e-4, betas=(0.5,0.999))

    conjugate = function.get_conjugate()
    output_activation = function.get_output_activation()

    for epoch in range(num_epochs):
        print('Epoch {}:'.format(epoch))
        for i,(X,y) in enumerate(dataloader):
            # Train T first
            X = X.to(device)
            optim_T.zero_grad()
            real_outputs = output_activation(T(X))
            noise = sampling(batch_size,100,device)
            fake_samples = gen(noise).detach()
            fake_outputs = conjugate(output_activation(T(fake_samples)))
            T_loss = torch.mean(fake_outputs)-torch.mean(real_outputs)
            T_loss.backward()
            optim_T.step()

            # Train G next
            optim_gen.zero_grad()
            fake_samples = gen(noise)
            fake_outputs = -conjugate(output_activation(T(fake_samples)))
            fake_loss = torch.mean(fake_outputs)
            fake_loss.backward()
            optim_gen.step()
            if i%50 == 0:
                print('T_loss: {}. Fake_loss: {}'.format(T_loss.data,fake_loss.data))
            
        imgs = gen(fix_noise)
        show_image(imgs,epoch)
    
    torch.save(gen.state_dict(),'pearson_mnist.pth')
    