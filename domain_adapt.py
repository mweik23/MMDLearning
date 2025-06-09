import numpy as np
import torch
from torch import nn

class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None, device=None):
        super().__init__()
        self.bandwidth_multipliers = (mul_factor**(torch.arange(n_kernels) - n_kernels // 2).float()).to(device)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            bwidth = L2_distances.sum() / (n_samples ** 2 - n_samples) #L2_distances.data is typically there
            #print('bwidth = ', bwidth)
            return bwidth

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        bwidth = self.get_bandwidth(L2_distances) * self.bandwidth_multipliers
        #print(bwidth)
        res = torch.exp(-L2_distances[None, ...]/bwidth[:, None, None]) #(np.pi*bwidth[:, None, None])**(-1/2)*
        #print('exp: ', res)
        #print('bwidth = ', bwidth[:, None, None])
        #print('kernel: ', res)
        return  res.sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))
        #print('memory of K: ', K.element_size() * K.nelement())
        #print('kernel: ', K)
        X_size = X.shape[0]
        Y_size = Y.shape[0]
        XX = (1/(1-1/X_size))*(K[:X_size, :X_size].mean() - K[0, 0]/X_size)
        XY = K[:X_size, X_size:].mean()
        YY = (1/(1-1/Y_size))*(K[X_size:, X_size:].mean() - K[0, 0]/Y_size)
        #print('memory of XX: ', XX.element_size())
        return XX - 2 * XY + YY

if __name__=='__main__':
    mmd = MMDLoss()
    n_data = 1000
    X = torch.normal(0, 100, size=(n_data, 2))
    Y = torch.normal(0, 100, size=(n_data, 2))
    res = mmd(X, Y)
    print(res)