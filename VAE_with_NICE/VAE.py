"""VAE model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import nice

class Model(nn.Module):
    def __init__(self, in_out_dim, mid_dim, device, coupling_type, coupling, hidden, latent_dim):
        """Initialize a VAE.

        Args:
            latent_dim: dimension of embedding
            device: run on cpu or gpu
        """
        super(Model, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 1, 2),   # B,  32, 28, 28
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 14, 14
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  7, 7
        )

        self.mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.logvar = nn.Linear(64 * 7 * 7, latent_dim)

        self.upsample = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),     # B,  64,  14,  14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1, 1),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 2, 4, 1, 2),      # B, 1, 28, 28
            nn.Sigmoid()
        )
        
        self.flow = nice.NICE(
                'gaussian',
                coupling,
                coupling_type,
                28*28,
                mid_dim,
                hidden,
                device=device).to(device)


    def sample(self,sample_size,mu=None,logvar=None,forward_=0,x=None):
        '''
        :param sample_size: Number of samples
        :param mu: z mean, None for prior (init with zeros)
        :param logvar: z logstd, None for prior (init with zeros)
        :return:
        '''
        if mu==None:
            mu = torch.zeros((sample_size,self.latent_dim)).to(self.device)
        if logvar == None:
            logvar = torch.zeros((sample_size,self.latent_dim)).to(self.device)

        sample_out = self.z_sample(mu, logvar)
        upsample_out = self.upsample(sample_out)
        upsample_out = torch.reshape(upsample_out,[sample_size, 64, 7, 7])
        decoder_out = self.decoder(upsample_out)
        
        mu_x,logvar_x = torch.split(decoder_out,1,dim=1)
            
        # flatt the matrix to vector 
        mu_x_vec = mu_x.view(sample_size, -1)
        logvar_x_vec = logvar_x.view(sample_size, -1) 
        sig_x =  torch.exp(0.5 * logvar_x_vec)
       
        
        if forward_:
            xT, total_log_prob = self.flow(x, mu_x_vec, sig_x)
            return total_log_prob
        else:
            xT = self.flow.sample(mu_x_vec,sig_x)
            x_hat = torch.reshape(xT,[sample_size, 1, 28, 28])
            return x_hat               


    def z_sample(self, mu, logvar):
        
        eps = torch.randn(mu.size()).to(self.device)  
        sig = torch.exp(0.5 * logvar)
        z = mu + sig * eps   
        return z       


    def loss(self,log_pxz,mu_z,logvar_z):

        # KL
        Ki_Li = 0.5 * (-logvar_z + torch.exp(logvar_z) + torch.square(mu_z) - 1)  #-0.5*(2log(sig)-sig^2-mu^2+1)
        KL = torch.sum(Ki_Li,dim = 1)  # KL(p||q) = sum(KL(pi||qi))
        
        ELBO = log_pxz - KL
        
        return -ELBO


    def forward(self, x):
        
        B, C, H, W = x.size()
        x_vec = x.view(B, -1)
        encoder_out = self.encoder(x)
        encoder_out = encoder_out.view(B, -1)    # flatt the matrix to vector 
        mu_out = self.mu(encoder_out)
        logvar_out = self.logvar(encoder_out)       
        log_pxz = self.sample(B,mu_out,logvar_out,1,x_vec)        
        
        ELBO  = self.loss(log_pxz,mu_out,logvar_out)
        
        return ELBO