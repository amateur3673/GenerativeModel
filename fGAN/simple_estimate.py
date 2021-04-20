import torch
import torch.nn as nn
import numpy as np

mean = 2
std = 1.5

def gen_real_samples(batch_size):
    return mean+std*torch.randn(batch_size,1)

def sampling(batch_size):
    return torch.randn(batch_size,1)

class Generator(nn.Module):
    def __init__(self,input_units = 1):
        super(Generator,self).__init__()
        self.linear = nn.Linear(input_units,input_units)
    
    def forward(self,x):
        return self.linear(x)

class T_function(nn.Module):
    def __init__(self,input_units = 1,hidden_units=64):
        super(T_function,self).__init__()
        self.linear1 = nn.Linear(input_units,hidden_units)
        self.linear2 = nn.Linear(hidden_units,hidden_units)
        self.linear3 = nn.Linear(hidden_units,1)
    
    def forward(self,x):
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = self.linear3(x)
        return torch.sigmoid(x)

class Estimate_function:
    def __init__(self,divergence_type):
        self.divergence_type = divergence_type
    def GAN_conjugate(self,x):
        return -torch.log(1-torch.exp(x))
    
    def KL_conjugate(self,x):
        return torch.exp(x-1)
    
    def KL_output_activation(self,x):
        return x
    
    def reverse_KL_conjugate(self,x):
        return -1-torch.log(-x)
    
    def reverse_KL_output_activation(self,x):
        return -torch.exp(x)
    
    def JS_conjugate(self,x):
        return -torch.log(2-torch.exp(x))
    
    def JS_activation(self,x):
        return np.log(2)-torch.log(1+torch.exp(-x))

    def GAN_output_activation(self,x):
        return -torch.log(1+torch.exp(-x))
    
    def pearson_conjugate(self,x):
        return x**2/4+x
    
    def pearson_output_activation(self,x):
        return x
    
    def get_conjugate(self):
        if self.divergence_type == 'GAN':
            return self.GAN_conjugate
        elif self.divergence_type == 'KL':
            return self.KL_conjugate
        elif self.divergence_type == 'reverse_KL':
            return self.reverse_KL_conjugate
        elif self.divergence_type == 'JS':
            return self.JS_conjugate
        elif self.divergence_type == 'pearson':
            return self.pearson_conjugate
            
    def get_output_activation(self):
        if self.divergence_type == 'GAN':
            return self.GAN_output_activation
        elif self.divergence_type == 'KL':
            return self.KL_output_activation
        elif self.divergence_type == 'reverse_KL':
            return self.reverse_KL_output_activation
        
        elif self.divergence_type == 'JS':
            return self.JS_activation
        elif self.divergence_type == 'pearson':
            return self.pearson_output_activation

if __name__=='__main__':
    # Get model
    
    gen = Generator()
    T = T_function()
    function = Estimate_function('GAN')
    conjugate = function.get_conjugate()
    output_activation = function.get_output_activation()
    gen.train()
    T.train()
    # datasets
    batch_size = 1024
    n_iters = 1000
    optim_gen = torch.optim.Adam(gen.parameters(),lr=0.01,betas=(0.5,0.999))
    optim_T = torch.optim.Adam(T.parameters(),lr = 0.01,betas=(0.5,0.999))
    for i in range(n_iters):
        print('Iteration {}: '.format(i),end='')
        # Train the T function
        optim_T.zero_grad()
        real_samples = gen_real_samples(batch_size)
        #real_outputs = output_activation(T(real_samples))
        real_outputs = T(real_samples)
        noise = sampling(batch_size)
        fake_samples = gen(noise)
        fake_outputs = conjugate(output_activation(T(fake_samples.detach())))
        T_loss = torch.mean(fake_outputs-real_outputs)
        T_loss.backward()
        optim_T.step()

        # Train gen
        optim_gen.zero_grad()
        fake_samples = gen(noise)
        fake_outputs = -conjugate(output_activation(T(fake_samples)))
        fake_loss = torch.mean(fake_outputs)
        fake_loss.backward()
        optim_gen.step()
        print('T_loss: {}. Fake_loss: {}'.format(T_loss.data,fake_loss.data))
    
    #torch.save(gen.state_dict(),'gen_GAN.pth')
    
    #torch.save(T_function.state_dict(),'T_GAN.pth')
    #gen = Generator()
    #gen.load_state_dict(torch.load('gen_GAN.pth'))
    #gen.eval()
    #print(gen.linear.weight,gen.linear.bias)