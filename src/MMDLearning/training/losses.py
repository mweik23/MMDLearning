import numpy as np
import torch
from torch import nn

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

    def __init__(self, n_feat=32, n_fourier=500, n_kernels=5, mul_factor=2.0, scale=None):
        super().__init__()
        #sample random frequencies
        omegas = sample_omegas(n_fourier, n_feat)  #shape (n_fourier, n_feat)
        self.register_buffer("omegas", omegas, persistent=True)

        #create bandwidth multipliers
        multipliers = (mul_factor ** (torch.arange(n_kernels) - n_kernels // 2).float())
        self.register_buffer("bandwidth_multipliers", multipliers, persistent=True) 
        self.scale = scale
        
    def get_scale(self, L2_distances):
        if self.scale is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.sum() / (n_samples ** 2 - n_samples)
        return self.scale

    def forward(self, X, Y):
        pass

def sample_omegas(n_fourier, n_feat, method='indep'):
    if method=='indep':
        return torch.normal(0, 1, size=(n_fourier, n_feat))
    elif method=='orthog':
        pass

if __name__=='__main__':
    mmd = MMDLoss()
    n_data = 1000
    X = torch.normal(0, 100, size=(n_data, 2))
    Y = torch.normal(0, 100, size=(n_data, 2))
    res = mmd(X, Y)
    print(res)