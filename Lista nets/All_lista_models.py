from torch import nn
import torch
#torch.set_default_dtype(torch.float)
# if torch.cuda.is_available():
#     torch.set_default_device('cuda')
# else:
#     torch.set_default_device('cpu')


class soft_thresh_module_ss(nn.Module):

    def __init__(self, n, alpha=0.001, learnable_alpha=False, ss_size = None):
        super().__init__()

        self.ss_size = ss_size
        self.n = n
        self.threshold = torch.nn.Parameter(torch.ones((self.n,))*alpha)
        if not learnable_alpha:
            self.threshold.requires_grad = False

        self.zero = torch.tensor([0.0])
        if self.ss_size is not None:
            self.topk = int(self.ss_size*self.n)

    def forward(self, x):
        x_abs = torch.abs(x)
        vals = torch.sign(x) * torch.maximum(x_abs - self.threshold, self.zero)

        if self.ss_size:
            sort_idx = torch.topk(x_abs, self.topk).indices
            idx = torch.zeros_like(x).scatter_(1, sort_idx, 1).to(torch.bool)
            vals[idx] = x[idx]

        return vals



# def soft_thresh_torch(x, l, zero=torch.tensor([0.0])):
#     return torch.sign(x) * torch.maximum(torch.abs(x) - l, zero)

class soft_thresh_module(nn.Module):

    def __init__(self, n, alpha=0.001, learnable_alpha=False):
        super().__init__()

        self.threshold = torch.nn.Parameter(torch.ones((n,))*alpha)
        if not learnable_alpha:
            self.threshold.requires_grad = False
        self.zero = torch.tensor([0.0])

    def forward(self, x):
        return torch.sign(x) * torch.maximum(torch.abs(x) - self.threshold, self.zero)


class baseline(nn.Module):

    def __init__(self, n=784, T=3, alpha=0.001, learnable_alpha=False, bias=True, ss_size=None):
        super().__init__()

        blocks = []
        for _ in range(T):
            blocks.append(nn.Linear(in_features=n, out_features=n, bias=bias))
            blocks.append(soft_thresh_module_ss(n, alpha=alpha, learnable_alpha=learnable_alpha, ss_size=ss_size))

        self.MLP = nn.Sequential(*blocks)

    def forward(self, x):
        return self.MLP(x)

class baseline_with_skipcon(nn.Module):

    def __init__(self, n=784, T=3, alpha=0.001, learnable_alpha=False, bias=True, ss_size=None):
        super().__init__()

        self.T = T
        self.W1 = []
        self.S = []
        for _ in range(self.T):
            self.W1.append(nn.Linear(in_features=n, out_features=n, bias=bias))
            self.S.append(soft_thresh_module_ss(n, alpha=alpha, learnable_alpha=learnable_alpha, ss_size=ss_size))

        #self.MLP = nn.Sequential(*zip(self.blocks_W, self.blocks_ST))
        self.W1 = nn.ParameterList(self.W1)
        self.S = nn.ParameterList(self.S)

    def forward(self, x):
        x = self.S[0](self.W1[0](x))
        for i in range(1, self.T):
            x = self.S[i](x + self.W1[i](x))

        return x

class lista_paper(nn.Module):

    def __init__(self,
                 n=784,
                 alpha=0.001,
                 learnable_alpha=False,
                 T=5,
                 bias=True,
                 ss_size=None):
        super().__init__()

        self.T = T
        self.W = nn.Linear(in_features=n, out_features=n, bias=bias)
        self.S = nn.Linear(in_features=n, out_features=n, bias=bias)
        self.sth = soft_thresh_module_ss(n, alpha=alpha, learnable_alpha=learnable_alpha, ss_size=ss_size)

    def forward(self, x):
        B = self.W(x)
        Zk = self.sth(B)
        for _ in range(1, self.T):
            Ct = B + self.S( Zk )
            Zk = self.sth( Ct )

        return Zk


class lista_wk_sk(nn.Module):

    def __init__(self,
                 n=784,
                 alpha=0.001,
                 learnable_alpha=False,
                 T=5,
                 bias=True,
                 ss_size=None):
        super().__init__()

        self.T = T
        self.W1 = [nn.Linear(in_features=n, out_features=n, bias=bias) for _ in range(self.T)]
        self.W2 = [nn.Linear(in_features=n, out_features=n, bias=bias) for _ in range(self.T)]
        self.S = [soft_thresh_module_ss(n, alpha=alpha, learnable_alpha=learnable_alpha, ss_size=ss_size)
                  for _ in range(self.T)]

        self.W1 = nn.ParameterList(self.W1)
        self.W2 = nn.ParameterList(self.W2)
        self.S = nn.ParameterList(self.S)

    def forward(self, x):
        z_k = self.W1[0](x)
        z_k = self.S[0](z_k)

        for i in range(1, self.T):
            z_k = self.W1[i](x) + self.W2[i](z_k)
            z_k = self.S[i](z_k)

        return z_k

class lista_cp_wk(nn.Module):

    def __init__(self,
                 A,
                 n=784,
                 alpha=0.001,
                 learnable_alpha=False,
                 T=5,
                 bias=True,
                 ss_size=None):

        super().__init__()

        self.n = n
        self.T = T
        self.A = torch.clone(A.T)


        self.W1 = [nn.Linear(in_features=self.n, out_features=self.n, bias=bias) for _ in range(self.T)]
        self.S = [soft_thresh_module_ss(n, alpha=alpha, learnable_alpha=learnable_alpha, ss_size=ss_size)
                  for _ in range(self.T)]

        self.W1 = nn.ParameterList(self.W1)
        self.S = nn.ParameterList(self.S)

    def forward(self, x):
        z_k = self.W1[0](x)
        z_k = self.S[0](z_k)

        for i in range(1, self.T):
            z_k = z_k + self.W1[i](x - z_k @ self.A)
            z_k = self.S[i](z_k)

        return z_k



class lista_cp_w0(nn.Module):

    def __init__(self,
                 A,
                 n=784,
                 alpha=0.001,
                 learnable_alpha=False,
                 T=5,
                 bias=True,
                 ss_size=None):
        super().__init__()

        self.n = n
        self.T = T
        self.A = torch.clone(A.T)

        self.W = nn.Linear(in_features=self.n, out_features=self.n, bias=bias)
        self.S = [soft_thresh_module_ss(n, alpha=alpha, learnable_alpha=learnable_alpha, ss_size=ss_size)
                  for _ in range(self.T)]

        #self.W1 = nn.ParameterList(self.W1)
        self.S = nn.ParameterList(self.S)

    def forward(self, x):
        z_k = self.W(x)
        z_k = self.S[0](z_k)

        for i in range(1, self.T):
            z_k = z_k + self.W(x - z_k @ self.A)
            z_k = self.S[i](z_k)

        return z_k



class lista_w0_sk(nn.Module):

    def __init__(self, n=784, alpha=0.001, learnable_alpha=False, T=5, bias=True, ss_size=None):
        super().__init__()

        self.T = T
        self.W = nn.Linear(in_features=n, out_features=n, bias=bias)
        self.S = [nn.Linear(in_features=n, out_features=n, bias=bias) for _ in range(self.T)]
        self.S = nn.ParameterList(self.S)
        self.sth = [soft_thresh_module_ss(n, alpha=alpha, learnable_alpha=learnable_alpha, ss_size=ss_size)
                    for _ in range(self.T)]

    def forward(self, x):
        B = self.W(x)
        Zk = self.sth[0](B)
        for k in range(1, self.T):
            Ct = B + self.S[k]( Zk )
            Zk = self.sth[k]( Ct )

        return Zk


from numba import jit
from numba import float64
from numba import int64
import numpy as np


@jit((float64[:], int64), nopython=True, nogil=True)
def ewma(arr_in, window):
    r"""Exponentialy weighted moving average specified by a decay ``window``
    to provide better adjustments for small windows via:

        y[t] = (x[t] + (1-a)*x[t-1] + (1-a)^2*x[t-2] + ... + (1-a)^n*x[t-n]) /
               (1 + (1-a) + (1-a)^2 + ... + (1-a)^n).

    Parameters
    ----------
    arr_in : np.ndarray, float64
        A single dimenisional numpy array
    window : int64
        The decay window, or 'span'

    Returns
    -------
    np.ndarray
        The EWMA vector, same length / shape as ``arr_in``

    Examples
    --------
    #>>> import pandas as pd
    #>>> a = np.arange(5, dtype=float)
    #>>> exp = pd.DataFrame(a).ewm(span=10, adjust=True).mean()
    #>>> np.array_equal(_ewma_infinite_hist(a, 10), exp.values.ravel())
    True
    """
    n = arr_in.shape[0]
    ewma = np.empty(n, dtype=float64)
    alpha = 2 / float(window + 1)
    w = 1
    ewma_old = arr_in[0]
    ewma[0] = ewma_old
    for i in range(1, n):
        w += (1-alpha)**i
        ewma_old = ewma_old*(1-alpha) + arr_in[i]
        ewma[i] = ewma_old / w
    return ewma

