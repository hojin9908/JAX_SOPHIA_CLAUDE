import numpy as np
import torch

eps = 1e-6

def gauss2d_kernel(xi, xj, h):
    r = np.linalg.norm(xi-xj) + eps
    alpha = (1/(np.pi*(h**2)))
    exp_term = np.exp(-(r/h)**2)
    domain = r < 3*h
    return alpha*exp_term*domain


def grad2d_gauss_kernel(xi, xj, h):
    r_vec = xi - xj
    r = np.linalg.norm(xi-xj) + eps
    alpha = (1/(np.pi*(h**2)))
    exp_term = np.exp(-(r/h)**2)
    grad = -2 * alpha * exp_term * r_vec / h**2
    domain = r < 3*h
    return  grad*domain


def pres2d_kernel(xi, xj, h):
    r = np.linalg.norm(xi-xj) + eps
    alpha = 10/np.pi
    term = (h - r)**3
    domain = r < 3*h
    return alpha*term*domain


def grad2d_pres_kernel(xi, xj, h):
    r_vec = xi - xj
    r = np.linalg.norm(xi-xj) + eps
    alpha = -30/np.pi
    term = (h - r)**2
    grad = alpha * term * r_vec / r
    domain = r < 3*h
    return grad*domain


def Wendland2_2d_kernel(xi, xj, h):
    r = np.linalg.norm(xi-xj)
    const = 7 / (np.pi*(4*h**2))
    term = ((1 - r/(2*h))**4)*(1 + 4*r/(2*h))
    domain = r < 2*h
    return const*term*domain
    

def grad2d_Wendland2_kernel(xi, xj, h):
    r_vec = xi - xj
    r = np.linalg.norm(xi-xj)
    const = 7 / (np.pi*(4*h**2))
    term = -20 * (1 - r/(2*h))**3 * (r/(2*h))
    grad = const/(2*h)*term * r_vec/(r+eps)
    domain = r < 2*h
    return grad*domain


################# CUDA version #################


def torch_gauss2d_kernel(xi, xj, h):
    r = torch.linalg.norm(xi-xj, dim=-1) + eps
    alpha = (1/(torch.pi*(h**2)))
    exp_term = torch.exp(-(r/h)**2)
    domain = r < 3*h
    return alpha*exp_term*domain


def torch_grad2d_gauss_kernel(xi, xj, h):
    r_vec = xi - xj
    r = torch.linalg.norm(xi-xj, dim=-1) + eps
    r = torch.stack((r,r), dim=-1)
    alpha = (1/(torch.pi*(h**2)))
    exp_term = torch.exp(-(r/h)**2)
    grad = -2 * alpha * exp_term * r_vec / h**2
    domain = r < 3*h
    return  grad*domain


def torch_pres2d_kernel(xi, xj, h):
    r = torch.linalg.norm(xi-xj, dim=-1) + eps
    alpha = 10/torch.pi
    term = (h - r)**3
    domain = r < 3*h
    return alpha*term*domain


def torch_grad2d_pres_kernel(xi, xj, h):
    r_vec = xi - xj
    r = torch.linalg.norm(xi-xj, dim=-1) + eps
    r = torch.stack((r,r), dim=-1)
    alpha = -30/torch.pi
    term = (h - r)**2
    grad = alpha * term * r_vec / r
    domain = r < 3*h
    return grad*domain


def torch_Wendland2_2d_kernel(xi, xj, h):
    r = torch.linalg.norm(xi-xj, dim=-1)        # [#sub, #neig]
    const = 7 / (torch.pi*(4*h**2))
    term = ((1 - r/(2*h))**4)*(1 + 4*r/(2*h))   # [#sub, #neig]
    domain = r < 2*h
    return const*term*domain
    

def torch_grad2d_Wendland2_kernel(xi, xj, h):
    r_vec = xi - xj                             # [#sub, #neig, 2]
    r = torch.linalg.norm(xi-xj, dim=-1)        # [#sub, #neig]
    r = torch.stack((r,r), dim=-1)              # [#sub, #neig, 2]
    const = 7 / (torch.pi*(4*h**2))
    term = -20 * (1 - r/(2*h))**3 * (r/(2*h))
    grad = const/(2*h)*term * r_vec/(r+torch.Tensor(r==0)+eps)
    domain = r < 2*h
    return grad*domain