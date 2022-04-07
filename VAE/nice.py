"""NICE model
"""

import torch
import torch.nn as nn
from torch.distributions.transforms import Transform,SigmoidTransform,AffineTransform
from torch.distributions import Uniform, TransformedDistribution
import numpy as np

"""Additive coupling layer.
"""
class AdditiveCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize an additive coupling layer.

        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AdditiveCoupling, self).__init__()
        
        Coupling_layers = []
        Coupling_layers.append(nn.Linear(in_out_dim//2, mid_dim))
        Coupling_layers.append(nn.ReLU())
        for i in range(hidden - 1):
            Coupling_layers.append(nn.Linear(mid_dim, mid_dim))
            Coupling_layers.append(nn.ReLU())
        Coupling_layers.append(nn.Linear(mid_dim, in_out_dim//2))
        self.Coupling_layers = nn.Sequential(*Coupling_layers) 
        self.mask_config = mask_config

    def forward(self, x, log_det_J, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """
        
        L, W = x.size()
        x = x.reshape((L, W//2, 2))
        if self.mask_config:
            xI1, xI2 = x[:, :, 1], x[:, :, 0]
        else:
            xI2, xI1 = x[:, :, 1], x[:, :, 0]

        y1 = xI1
        t = self.Coupling_layers(xI1)
        if not reverse:
            y2 = xI2 + t
        else:
            y2 = xI2 - t

        if self.mask_config:
            x = torch.stack((y2, y1), dim=2)
        else:
            x = torch.stack((y1, y2), dim=2)
            
        return x.reshape((L, W)), log_det_J
    
    
class AffineCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize an affine coupling layer.

        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AffineCoupling, self).__init__()

        Coupling_layers = []
        Coupling_layers.append(nn.Linear(in_out_dim//2, mid_dim))
        Coupling_layers.append(nn.ReLU())
        for i in range(hidden - 1):
            Coupling_layers.append(nn.Linear(mid_dim, mid_dim))
            Coupling_layers.append(nn.ReLU())
        Coupling_layers.append(nn.Linear(mid_dim, in_out_dim//2))
        self.Coupling_layers = nn.Sequential(*Coupling_layers) 
        self.tanh_ = nn.Tanh()
        self.mask_config = mask_config
        

    def forward(self, x, log_det_J, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """
        
        L, W = x.size()
        x = x.reshape((L, W//2, 2))
        if self.mask_config:
            xI1, xI2 = x[:, :, 1], x[:, :, 0]
        else:
            xI2, xI1 = x[:, :, 1], x[:, :, 0]
   
        y1 = xI1
        t = self.Coupling_layers(xI1)
        s = self.tanh_(t)
        
        if not reverse:
            y2 = torch.exp(s) * xI2 + t
            log_det_J += torch.sum(s, dim=1)
        else:
            y2 = torch.exp(-s) * (xI2 - t)
 
        if self.mask_config:
            x = torch.stack((y2, y1), dim=2)
        else:
            x = torch.stack((y1, y2), dim=2)
            
        return x.reshape((L, W)), log_det_J
        

"""Log-scaling layer.
"""
class Scaling(nn.Module):
    def __init__(self, dim):
        """Initialize a (log-)scaling layer.

        Args:
            dim: input/output dimensions.
        """
        super(Scaling, self).__init__()
        self.scale = nn.Parameter(
            torch.zeros((1, dim)), requires_grad=True)
        self.eps = 1e-5

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log-determinant of Jacobian.
        """
        scale = torch.exp(self.scale)+ self.eps
        
        log_det_J = torch.sum(scale)
        if not reverse:
            x = x * torch.exp(scale)
        else:
            x = x * torch.exp(-scale)
        return x, log_det_J
               

"""Standard logistic distribution.
"""
logistic = TransformedDistribution(Uniform(0, 1), [SigmoidTransform().inv, AffineTransform(loc=0., scale=1.)])

"""NICE main model.
"""
class NICE(nn.Module):
    def __init__(self, prior, coupling, coupling_type,
        in_out_dim, mid_dim, hidden,device):
        """Initialize a NICE.

        Args:
            coupling_type: 'additive' or 'adaptive'
            coupling: number of coupling layers.
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            device: run on cpu or gpu
        """
        super(NICE, self).__init__()
        self.device = device
        if prior == 'gaussian':
            self.prior = torch.distributions.Normal(
                torch.tensor(0.).to(device), torch.tensor(1.).to(device))
        elif prior == 'logistic':
            self.prior = logistic
        else:
            raise ValueError('Prior not implemented.')
        self.in_out_dim = in_out_dim
        self.coupling = coupling
        self.coupling_type = coupling_type
        
        if coupling_type == 'additive':
            self.Coupling_l = nn.ModuleList([AdditiveCoupling(in_out_dim, mid_dim, hidden, mask_config = i%2) for i in range(coupling)])
        else:
            self.Coupling_l = nn.ModuleList([AffineCoupling(in_out_dim, mid_dim, hidden, mask_config = i%2) for i in range(coupling)])
        
        self.Scaling_ = Scaling(in_out_dim)
               

    def f_inverse(self, z):
        """Transformation g: Z -> X (inverse of f).

        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """

        x, _ = self.Scaling_(z, reverse = True)
        log_det_J = 0
        for i in reversed(range(self.coupling)):
            x, log_det_J = self.Coupling_l[i](x,log_det_J, reverse=True)
        return x 
    

    def f(self, x):
        """Transformation f: X -> Z (inverse of g).

        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z and log determinant Jacobian
        """
        
        log_det_J = 0
        for i in range(self.coupling):
            x, log_det_J = self.Coupling_l[i](x, log_det_J, reverse=False)
        z, log_det_J_scal = self.Scaling_(x, reverse = False)
        
        return z, log_det_J + log_det_J_scal 
        

    def log_prob(self, x, mu, sig):
        """Computes data log-likelihood.

        (See Section 3.3 in the NICE paper.)

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        z, log_det_J = self.f(x)                        # x_original_image
        log_det_J -= np.log(256)*self.in_out_dim        #log det for rescaling from [0.256] (after dequantization) to [0,1]
        normal_dis = torch.distributions.Normal(mu, sig)
        log_ll = torch.sum(normal_dis.log_prob(z), dim=1) #mu, sig decoder 
        return z, log_ll + log_det_J

    def sample(self, mu, sig):
        """Generates samples.

        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """
        normal_dis = torch.distributions.Normal(mu, sig)
        z = normal_dis.sample()
        
        return self.f_inverse(z)

    def forward(self, x, mu, sig):
        """Forward pass.

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        
        z,loss = self.log_prob(x, mu, sig)
        
        return z,loss