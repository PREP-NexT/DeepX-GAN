import torch
import scipy.sparse
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from libpysal.weights import lat2W
import pandas as pd


def crs_to_torch_sparse(x):
    """
        Convert sparse scipy matrix to torch sparse tensor.
    
    :param x: crs matrix (scipy sparse matrix)
    :return: weight matrix as torch sparse tensor
    """
    x = x.tocoo()
    values = x.data
    indices = np.vstack((x.row, x.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = x.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def make_sparse_weight_matrix(h, w, rook=False):
    """
        Create a spatial weight matrix using libpysal's lat2W function.
        
    :param h: height of the matrix
    :param w: width of the matrix
    :param rook: whether to use rook weights or not
    :return: w = weight matrix as torch sparse tensor
    """
    w = lat2W(h, w, rook=rook)      # For rook, the neighboring pixels are weight 1's; for queen, the neighboring
                                    # (including clinodiagonal) pixels are weight 1's.
    return crs_to_torch_sparse(w.sparse)


def temporal_weights(n,b):
    """
        Compute temporal distance weight tensor.
        
    :param n: number of time steps
    :param b: parameter governing exponential weight decay
    :return: weights = temporal weights tensor of shape [1, 1, n-1]
    """
    weights = torch.exp(-torch.arange(1,n).flip(0) / b).view(1,1,-1)
    return weights


def get_tdc(a, u=0.8, correct_diag=True):
    """
        Pytorch version to compute tail dependence coefficient matrix, vectorized by matrix multiplication

    :param a: [height*width, n_frames], torch.tensor
    :param u: threshold, default=0.8
    :param correct_diag: to ensure diag=1
    :return: probs - torch.tensor, tail dependence coefficient matrix,
                    symmetric [height*width, height*width]
    """
    n, hw = a.shape
    in_tail = a > torch.quantile(a, q=u, dim=0)
    ## list comprehension
    # probs = [in_tail[in_tail[:, i], j].sum()/(n*(1-u)) for i in range(hw) for j in range(hw)]
    # probs = torch.stack(probs).reshape(hw, hw)
    ## vectorization
    probs = torch.matmul(in_tail.t().type(torch.float), in_tail.type(torch.float))
    probs = probs/(n*(1-u))
    if correct_diag:
        probs.fill_diagonal_(1)
    return probs


def paired_multiply(mat):
    """
        Obtain the multiplication between any two elements of a 2-D matrix. If the original matrix is [h, w], this multiplication will result in a [h*w, h*w] symmetric matrix,
    with each row being [xn*x1, xn*x2, xn*x3, ...].

    :param mat: matrix to be conducted. torch.tensor
    :return: mat_aug: Augmented matrix.
    """
    h, w = mat.shape
    mat_long = torch.reshape(mat, (h*w, 1))
    mat_rep = mat_long.repeat(1, h*w)
    mat_aug = torch.mul(mat_rep.type(torch.float), mat_rep.t().type(torch.float))
    return mat_aug


def get_weights_tdc(x, u=0.8):
    """
        Compute the tail dependence coefficient (TDC) matrix

    :param x: [height, width, n_frames], torch.tensor (n_frames = n_time_steps ?)
    :param u: threshold, default=0.8
    :return: weights_tdc [height, width, height*width]
    """
    h, w, n = x.shape
    x = torch.reshape(x, (h*w, n)).permute(1, 0)
    weights_tdc = get_tdc(x, u, correct_diag=True)
    weights_tdc = torch.reshape(weights_tdc, (h, w, h*w))
    # print("weights_tdc.is_cuda: ", weights_tdc.is_cuda)  # the output should be cuda (i.e.,GPU) if GPU is enabled
    # print("x.is_cuda: ", x.is_cuda)
    return weights_tdc


def get_weights_tdc_masked(x, u=0.8):
    """
        mask weights_tdc by (x_it > percentile(x_i{t}, u)) & (x_jt > percentile(x_j{t}, u))

    :param x: [height, width, n_frames], torch.tensor (n_frames = n_time_steps ?)
    :param u: the percentile threshold for extremes 
    :return: weights_tdc_masked [height, width, height*width, n_time_steps]
    """
    h, w, n = x.shape

    ## Compute the mask for weights_tdc (test whether both elements > percentile)
    mask_ele = x > torch.quantile(x, u, dim=2, keepdim=True)  ## torch.quantile: [height, width, 1], mask_ele: [height, width, n_ts]
    mask_paired = [paired_multiply(mask_ele[:, :, t]) for t in range(0, n)]   
    mask_paired = torch.stack(mask_paired)   ## mask_paired after stack: [n, h*w, h*w]
    ## Compute weights_tdc as usual
    x = torch.reshape(x, (h*w, n)).permute(1, 0)
    weights_tdc = get_tdc(x, u, correct_diag=True)  ## weights_tdc: [h*w, h*w]
    ## Mask weights_tdc
    weights_tdc_masked = torch.mul(weights_tdc, mask_paired)   ## broadcast automatically
    ## Reshape similar to that without mask
    weights_tdc_masked = weights_tdc_masked.permute(1, 2, 0)
    weights_tdc_masked = torch.reshape(weights_tdc_masked, (h, w, h*w, n))
    return weights_tdc_masked  ## [height, width, height*width, n_time_steps]

## mask_paired_sp = mask_paired.to_sparse()


def st_ex_tdc(x, weights, weights_tdc, mask_on=False):
    """
        Space-time expectations using the tail dependence coefficient (TDC) matrix, vectorized by matrix multiplication.
    
    :param x: input video of shape [height, width, n_frames]
    :param weights: tensor of distance weights of shape [1, 1, time_steps] (Can be computed via temporal_weights())
    :param weights_tdc: tail dependence coefficient [height, width, height*width]
    :param mask_on: whether to use the mask for TDC
    :return: exp_val = expected values;
             shape [height, width, n_frames-1]
    """
    h, w, n = x.shape
    # first term
    if mask_on:
        # x_aug = x.reshape(h,w,1,n)
        # term1 = [(weights_tdc[:, :, i, t] * x_aug[:, :, 0, t]).reshape(-1).sum() for i in range(h*w) for t in range(1, n)]

        weights_tdc_resize = weights_tdc.view(h*w, h*w, n)
        x_resize = x.reshape(h*w, n)
        term1 = [torch.matmul(weights_tdc_resize[:, :, t].t(), x_resize[:, 1:])[:, t-1] for t in range(1, n)]
        term1 = torch.stack(term1).t()
    else:
        # term1 = [(weights_tdc[:, :, i] * x[:, :, t]).reshape(-1).sum() for i in range(h*w) for t in range(1, n)]
        term1 = torch.matmul(weights_tdc.view(h*w, -1).t(), x.view(h*w, n)[:, 1:])  ## whichever is the first matrix need to be transposed, this is because the view/reshape puts the original [h,w] dimension to the h*w length column.

    # term1 = torch.stack(term1).reshape(h*w, n-1)        # [height*width, n_frames-1]
    exp_val = [(weights[:, :, -t:] * x[:, :, :t]).sum(dim=2).reshape(-1) *
               term1[:, t-1] / (weights[:, :, -t:] * x[:, :, :t]).reshape(-1).sum() for t in range(1, n)]
    exp_val = torch.stack(exp_val).permute(1, 0).reshape(h, w, n - 1)
    return exp_val


def handle_tuple_err(G):
    """
    (optional) Handle errors related to tuple attributes in the neural network layers.

    :param G: Generator model
    :return: G with handled tuple attributes
    """
    if hasattr(G, 'deconv_net'):
        for i in range(len(G.deconv_net)):
            if hasattr(G.deconv_net[i], 'kernel_size'):
                G.deconv_net[i].kernel_size = tuple(G.deconv_net[i].kernel_size)
            if hasattr(G.deconv_net[i], 'stride'):
                G.deconv_net[i].stride = tuple(G.deconv_net[i].stride)
            if hasattr(G.deconv_net[i], 'padding'):
                G.deconv_net[i].padding = tuple(G.deconv_net[i].padding)
    if hasattr(G, 'conv_net'):
        for i in range(len(G.conv_net)):
            if hasattr(G.conv_net[i], 'kernel_size'):
                G.conv_net[i].kernel_size = tuple(G.conv_net[i].kernel_size)
            if hasattr(G.conv_net[i], 'stride'):
                G.conv_net[i].stride = tuple(G.conv_net[i].stride)
            if hasattr(G.conv_net[i], 'padding'):
                G.conv_net[i].padding = tuple(G.conv_net[i].padding)
    return G


def st_ex(x, weights):
    """
        Space-time expectations (assuming temporal order to suit sequentiality constraints)
    
    :param x: input video of shape [height, width, n_frames]
    :param weights: tensor of distance weights of shape [1, 1, time_steps] (Can be computed via temporal_weights())
    :return: exp_val = expected values assuming space-time independence and sequential calculation;
             shape [height, width, n_frames-1]
    """
    h, w, n = x.shape
    exp_val = [(weights[:, :, -t:] * x[:, :, :t]).sum(dim=2).reshape(-1) * x[:, :, t].reshape(-1).sum() / (weights[:, :, -t:] * x[:, :, :t]).reshape(-1).sum() for t in range(1, n)]
    exp_val = torch.stack(exp_val).permute(1, 0).reshape(h, w, n - 1)
    return exp_val


def st_ex_kulldorff(x):
    """
        Space-time expectations; as proposed by Kulldorff, 2005 (assuming knowledge of the whole time series; no temporal weights)
        
    :param x: input video of shape [height, width, n_frames]
    :return: exp_val = expected values assuming space-time independence and sequential calculation;
             shape [height, width, n_frames-1]
    """
    h, w, n = x.shape
    s_ex = torch.stack([x[:, :, t].reshape(-1).sum() for t in range(0, n)])
    t_ex = x.sum(dim=2)
    exp_val = torch.einsum('ab,c->abc', (t_ex, s_ex)) / x.reshape(-1).sum()
    return exp_val


def st_ex_kulldorff_weighted(x, weights):
    """
        Space-time expectations; as proposed by Kulldorff, 2005 (assuming knowledge of the whole time series; including temporal weights)
    
    :param x: input video of shape [height, width, n_frames]
    :param weights: tensor of distance weights of shape [1, 1, time_steps] (Can be computed via temporal_weights())
    :return: exp_val = expected values assuming space-time independence and sequential calculation;
             shape [height, width, n_frames-1]
    """
    h, w, n = x.shape
    exp_val = torch.stack([x[:, :, t].reshape(-1).sum() * (x * weights[t,...].reshape(1, 1, -1)).sum(dim=2) / (x * weights[t,...].reshape(1,1,-1)).reshape(-1).sum() for t in range(0,n)]).permute(1,2,0)
    return exp_val


def mi_mean(x, x_mean, w_sparse):
    """
        Local Moran's I with custom means

    :param x: input data tensor (flattened or image)
    :param x_mean: input tensor of same (flattened) shape as x
    :param w_sparse: spatial weight matrix; torch sparse tensor 
    :return: mi = output data - local Moran's I
    """
    x = x.reshape(-1)
    n = len(x)
    n_1 = n - 1
    z = x - x_mean
    sx = x.std()
    z /= sx
    den = (z * z).sum()
    zl = torch.sparse.mm(w_sparse, z.reshape(-1, 1)).reshape(-1)
    mi = n_1 * z * zl / den
    return mi


def mi(x, w_sparse):
    """
        Local Moran's I
    
    :param x: input data tensor (flattened or image)
    :param w_sparse: spatial weight matrix; torch sparse tensor
    :return: mi = output data - local Moran's I    
    """
    x = x.reshape(-1)
    n = len(x)
    n_1 = n - 1
    z = x - x.mean()
    sx = x.std()
    z /= sx
    den = (z * z).sum()
    zl = torch.sparse.mm(w_sparse, z.reshape(-1, 1)).reshape(-1)
    mi = n_1 * z * zl / den
    return mi


def vid_mi(x, w_sparse):
    """
        Local Moran's I for a video (time-series of images)

    :param x: input video of shape [height, width, n_frames]
    :param w_sparse: spatial weight matrix; torch sparse tensor
    :return: mis = output data - local Moran's Is
    """
    h, w, n = x.shape
    mis = torch.stack([mi(x[:, :, i].reshape(-1), w_sparse).reshape(h, w) for i in range(n)])
    return mis


def make_mis(x, w_sparse):
    """
        Make Local Moran's I for a batch of videos
    
    :param x: input video batch of shape [batch_size, time_steps, n_channel, height, width]
    :param w_sparse: spatial weight matrix; torch sparse tensor
    :return: mis = output data - local Moran's I
    """
    n, t, nc, h, w = x.shape
    mis = torch.stack([vid_mi(x[i, :, j, :, :].reshape(t, h, w).permute(1, 2, 0), w_sparse) for j in range(nc) for i in range(n)]).reshape(n, t, nc, h, w)
    mis = torch.stack([(mis[i, :, j, :, :] - torch.min(mis[i, :, j, :, :])) / (torch.max(mis[i, :, j, :, :]) - torch.min(mis[i, :, j, :, :])) for j in range(nc) for i in range(n)]).reshape(n, t, nc, h, w)
    return mis


def spate(x, w_sparse, b, method="skw", b_tdc=None):
    """
        Space-time expectations (SPATE or DeepX if method="tdc_masked") for video data, using Local Moran's I.
    
    :param x: input video of shape [height, width, n_frames]
    :param w_sparse: spatial weight matrix; torch sparse tensor
    :param b: tensor of distance weights of shape [1, 1, time_steps] (Can be computed via temporal_weights())
    :param method: method to use for computing space-time expectations; default 'skw'
                (Options are sequential Kulldorff-weighted ('skw'), Kulldorff ('k'), Kulldorff-weighted ('kw'),
                 Tail Dependence Coefficient ('tdc'), or Tail Dependence Coefficient masked ('tdc_masked'))
    :param b_tdc: tensor of distance weights for TDC (can be computed via get_weights_tdc_masked())
    :return: spates = output data - SPATE or DeepX
    """
    h, w, n = x.shape
    if method == "k":
        x_means = st_ex_kulldorff(x)
    elif method == "kw":
        x_means = st_ex_kulldorff_weighted(x, b)
    ## added
    elif method == "tdc":
        x_means = st_ex_tdc(x, b, b_tdc, mask_on=False)
    elif method == "tdc_masked":
        x_means = st_ex_tdc(x, b, b_tdc, mask_on=True)
    else:
        x_means = st_ex(x, b)
    if (method=="skw") | (method == "tdc") | (method == "tdc_masked"):
        spates = torch.stack([mi_mean(x[:, :, i + 1].reshape(-1), x_means[:, :, i].reshape(-1), w_sparse).reshape(h, w) for i in range(n - 1)])
    else:
        spates = torch.stack([mi_mean(x[:, :, i].reshape(-1), x_means[:, :, i].reshape(-1), w_sparse).reshape(h, w) for i in range(n)])
    return spates.permute(1, 2, 0)


def make_spates(x, w_sparse, b, method="skw", u=0.8, theta1=0.5, theta2=0.5):
    """
        Make SPATEs(or, DeepX) for a batch of videos
    
    :param x: input video batch of shape [batch_size, time_steps, n_channel, height, width]
    :param w_sparse: spatial weight matrix; torch sparse tensor
    :param b: tensor of distance weights of shape [1, 1, time_steps] (Can be computed via temporal_weights())
    :param method: method to use for computing space-time expectations; default 'skw'
                (Options are sequential Kulldorff-weighted ('skw'), Kulldorff ('k'), Kulldorff-weighted ('kw'))
    :param u: threshold, default=0.8, only used in 'tdc'
    :param theta1: parameters to be used in "tdc_masked", the proportion of original SPATE in the combined metric
    :param theta2: parameters to be used in "tdc_masked", the proportion of new masked DeepX in the combined metric
    :return: output data - SPATE
    """
    n, t, nc, h, w = x.shape
    if method == "skw":
        spates = torch.stack([spate(x[i, :, j, :, :].reshape(t, h, w).permute(1, 2, 0), w_sparse, b, method).permute(2, 0, 1) for j in range(nc) for i in range(n)]).reshape(n, t - 1, nc, h, w)
        spates = torch.stack([(spates[i, :, j, :, :] - torch.min(spates[i, :, j, :, :])) / (torch.max(spates[i, :, j, :, :]) - torch.min(spates[i, :, j, :, :])) for j in range(nc) for i in range(n)]).reshape(n, t - 1, nc, h, w)
        spates = F.pad(spates, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        spates = torch.roll(spates, 1, 1)
    ## added
    elif method == 'tdc':
        spates = torch.stack([spate(x[i, :, j, :, :].reshape(t, h, w).permute(1, 2, 0), w_sparse, b, method, get_weights_tdc(x[i, :, j, :, :].reshape(t, h, w).permute(1, 2, 0), u)).permute(2, 0, 1) for j in range(nc) for i in range(n)]).reshape(n, t - 1, nc, h, w)
        spates = torch.stack([(spates[i, :, j, :, :] - torch.min(spates[i, :, j, :, :])) / (torch.max(spates[i, :, j, :, :]) - torch.min(spates[i, :, j, :, :])) for j in range(nc) for i in range(n)]).reshape(n, t - 1, nc, h, w)
        spates = F.pad(spates, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        spates = torch.roll(spates, 1, 1)
    ## added
    elif method == 'tdc_masked':
        spates_masked = torch.stack([spate(x[i, :, j, :, :].reshape(t, h, w).permute(1, 2, 0), w_sparse, b, method, get_weights_tdc_masked(x[i, :, j, :, :].reshape(t, h, w).permute(1, 2, 0), u)).permute(2, 0, 1) for j in range(nc) for i in range(n)]).reshape(n, t - 1, nc, h, w)
        spates_original = torch.stack([spate(x[i, :, j, :, :].reshape(t, h, w).permute(1, 2, 0), w_sparse, b, method="skw").permute(2, 0, 1) for j in range(nc) for i in range(n)]).reshape(n, t - 1, nc, h, w)
        spates = theta1 * spates_original + theta2 * spates_masked
        spates = torch.stack([(spates[i, :, j, :, :] - torch.min(spates[i, :, j, :, :])) / (torch.max(spates[i, :, j, :, :]) - torch.min(spates[i, :, j, :, :])) for j in range(nc) for i in range(n)]).reshape(n, t - 1, nc, h, w)
        spates = F.pad(spates, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        spates = torch.roll(spates, 1, 1)
    else:
        spates = torch.stack([spate(x[i, :, j, :, :].reshape(t, h, w).permute(1, 2, 0), w_sparse, b, method).permute(2, 0, 1) for j in range(nc) for i in range(n)]).reshape(n, t, nc, h, w)
        spates = torch.stack([(spates[i, :, j, :, :] - torch.min(spates[i, :, j, :, :])) / (torch.max(spates[i, :, j, :, :]) - torch.min(spates[i, :, j, :, :])) for j in range(nc) for i in range(n)]).reshape(n, t, nc, h, w)
    return spates

