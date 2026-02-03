import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        multipliers = (mul_factor ** (torch.arange(n_kernels) - n_kernels // 2).float())
        self.register_buffer("bandwidth_multipliers", multipliers, persistent=True)  # <- no device
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.sum() / (n_samples ** 2 - n_samples)
        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        # ensure buffer matches input
        mult = self.bandwidth_multipliers.to(dtype=X.dtype, device=X.device)
        bwidth = self.get_bandwidth(L2_distances)
        if not torch.is_tensor(bwidth):  # user passed a float
            bwidth = torch.as_tensor(bwidth, dtype=X.dtype, device=X.device)
        res = torch.exp(-L2_distances[None, ...] / (bwidth * mult)[:, None, None])
        return res.sum(dim=0)

class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))
        X_size = X.shape[0]
        Y_size = Y.shape[0]
        K.fill_diagonal_(0)
        XX = K[:X_size, :X_size].sum()/(X_size*(X_size-1))
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].sum()/(Y_size*(Y_size-1))
        return XX - 2 * XY + YY

class LambdaAdjust(nn.Module):

    def __init__(self, tau, kappa):
        super().__init__()
        self.register_buffer('tau', torch.as_tensor(tau))
        self.register_buffer('kappa', torch.as_tensor(kappa))
        self.relu  = nn.ReLU()

    def forward(self, mmd_val):
        # mmd_val is a scalar tensor
        if self.kappa==0:
            return torch.tensor(1.0, device=mmd_val.device)
        pos = self.relu(self.tau - mmd_val)
        return torch.exp(-self.kappa * pos)

class MMDScheduler:

    def __init__(self, turnon, width, coef=1):
        self.turnon=turnon
        self.width=width
        self.coef = coef

    def __call__(self, epoch):
        if self.width>0:
            return self.coef*(1+torch.tanh((torch.tensor(epoch)-self.turnon)/self.width))/2
        elif self.width==0:
            return self.coef*(epoch>=self.turnon)
        else:
            print('MMD scheduler width is not allowed to be negative')

class LinearizedMMDLoss(nn.Module):

    def __init__(self, n_feat=500, n_latent=32, n_kernels=5, mul_factor=2.0, scale=None, beta_k=None, gamma=2):
        super().__init__()
        #sample random frequencies
        omegas = sample_omegas(n_feat, n_latent)  #shape (n_feat, n_latent)
        self.n_feat = n_feat
        self.register_buffer("omegas", omegas, persistent=True)

        #create bandwidth multipliers (applied to sigma itself so sqrt of definition in RBF class)
        multipliers = (mul_factor ** ((torch.arange(n_kernels) - n_kernels // 2) / 2).float())
        self.register_buffer("bandwidth_multipliers", multipliers, persistent=True) 
        self.scale = scale
        self.beta_k = beta_k
        self.gamma = gamma
    def get_scale(self, X):
        if self.scale is None:
            #--------TODO: perform reduction from other ranks if ddp--------
            L2_distances = torch.cdist(X, X) ** 2
            n_samples = L2_distances.shape[0]
            val = L2_distances.sum() / (n_samples ** 2 - n_samples)
            return torch.sqrt(val.clamp_min(0))
            #---------------------------------------------------------------
        return self.scale
    
    def get_features(self, X, scale):
        proj = torch.einsum('ij, kj -> ik', X, self.omegas)[:, None, :] \
            / (scale*self.bandwidth_multipliers[None, :, None])
        feat = torch.cat(torch.sincos(proj), dim=-1)
        return feat
    
    def get_mmd_per_kernel(self, X, Y):
        X_size = X.shape[0]
        Y_size = Y.shape[0]
        scale = self.get_scale(torch.vstack([X, Y]))
        if not torch.is_tensor(scale):  # user passed a float
            scale = torch.as_tensor(scale, dtype=X.dtype, device=X.device)
        feat_X = self.get_features(X, scale)  #shape (n_x, n_kernels, 2*n_feat)
        feat_Y = self.get_features(Y, scale)  #shape (n_y, n_kernels, 2*n_feat)
        #---------TODO: perform reduction from other ranks if ddp---------
        sum_X = feat_X.sum(dim=0)  #shape (n_kernels, 2*n_feat)
        sum_Y = feat_Y.sum(dim=0)  #shape (n_kernels, 2*n_feat)
        #---------------------------------------------------------------
        #these take the mean over features sum over trig
        XX = ((sum_X * sum_X).sum(dim=-1) / self.n_feat - X_size) / (X_size * (X_size - 1))
        YY = ((sum_Y * sum_Y).sum(dim=-1) / self.n_feat - Y_size) / (Y_size * (Y_size - 1))
        XY = ((sum_X * sum_Y).sum(dim=-1) / self.n_feat) / (X_size * Y_size)
        return XX - 2 * XY + YY  #shape (n_kernels,)

    def forward(self, X, Y):
        mmd_per_kernel = self.get_mmd_per_kernel(X, Y)  #shape (n_kernels,)
        if self.beta_k is not None:
            soft_plus = self.beta_k * F.softplus(mmd_per_kernel/self.beta_k)
            return (soft_plus**self.gamma).sum()**(1/self.gamma)
        return mmd_per_kernel.sum()

def sample_omegas(n_feat, n_latent, method='indep'):
    if method=='indep':
        return torch.normal(0, 1, size=(n_feat, n_latent))
    elif method=='orthog':
        raise NotImplementedError("Orthogonal omega sampling not implemented yet")

if __name__=='__main__':
    mmd = MMDLoss()
    n_data = 1000
    X = torch.normal(0, 100, size=(n_data, 2))
    Y = torch.normal(0, 100, size=(n_data, 2))
    res = mmd(X, Y)
    print(res)