"""
models.py — Generator and Discriminator architectures for DeepX-GAN
======================================================================
This module defines the two neural network components of the GAN:

- **VideoDCG** (Video Deep Convolutional Generator): a spatiotemporal generator
  that maps Gaussian latent noise into synthetic sequences of gridded climate
  fields.  It uses two stacked LSTMs to capture temporal dynamics, followed by
  transposed-convolutional (deconvolutional) layers to upsample to the target
  spatial resolution.

- **VideoDCD** (Video Deep Convolutional Discriminator): a spatiotemporal
  discriminator that maps a real or generated sequence to a sequence of
  lower-dimensional feature vectors (h or M), used by the COT-GAN loss.
  It uses strided convolutional layers to extract spatial features at each
  time step, then two LSTM heads (H and M) to summarise temporal dynamics.

Architecture design choices follow the original COT-GAN paper (Xu et al. 2020)
with adaptations for non-square spatial domains (the generator padding is
computed analytically to hit the exact target resolution).

Reference
---------
Xu, T., et al. (2020). "COT-GAN: Generating Sequential Data via Causal
Optimal Transport." NeurIPS 2020.

[Paper citation here]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Discriminator (shared for D_H and D_M)
# ---------------------------------------------------------------------------

class VideoDCD(nn.Module):
    """
    Spatiotemporal discriminator.

    Given a sequence of spatial frames (real or generated), this network
    produces a sequence of feature vectors that are used by the COT-GAN loss
    as the H or M functionals.

    Architecture
    ------------
    1. Three strided 2-D convolutional layers (optionally with BatchNorm and
       LeakyReLU) extract spatial features from each frame independently.
    2. A shared LSTM encodes the temporal dynamics of the extracted features.
    3. Two parallel LSTM heads (final1, final2) produce the H and M outputs.

    Parameters
    ----------
    batch_size : int
        Mini-batch size (must match the size of the inputs at forward time).
    x_h : int
        Spatial height of each input frame.
    x_w : int
        Spatial width of each input frame.
    filter_size : int
        Number of feature maps in the first convolutional layer; doubled in
        each subsequent layer.
    j : int
        Output dimensionality of H and M feature vectors (J in the paper).
    nchannel : int
        Number of input channels.  When the DeepX embedding is concatenated,
        this is 2 (original field + embedding).
    bn : bool
        Whether to use BatchNorm after each conv/LSTM layer.
    """

    def __init__(
        self,
        batch_size: int,
        x_h: int = 64,
        x_w: int = 64,
        filter_size: int = 128,
        j: int = 16,
        nchannel: int = 1,
        bn: bool = False,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.filter_size = filter_size
        self.nchannel = nchannel
        self.j = j
        self.bn = bn
        self.x_height = x_h
        self.x_width = x_w

        # ---- Convolutional padding ----------------------------------------
        # We want the spatial resolution to halve at each conv layer so that
        # the LSTM input dimension is independent of the target resolution.
        ks = 6      # kernel size
        s = 2       # stride

        def conv_out_shape(h_in, p, k, st):
            return int(torch.floor(torch.tensor((h_in + 2.0 * p - k) / st)).item()) + 1

        def conv_padding(h_in, s, k_size):
            return max((h_in * (s - 2) - s + k_size) // 2, 0)

        p = conv_padding(8, s, ks)
        k_size = [ks, ks]
        stride = [s, s]
        padding = [p, p]
        input_shape = [x_h, x_w]

        out1 = [conv_out_shape(input_shape[0], p, ks, s), conv_out_shape(input_shape[1], p, ks, s)]
        out2 = [conv_out_shape(out1[0], p, ks, s), conv_out_shape(out1[1], p, ks, s)]
        out3 = [conv_out_shape(out2[0], p, ks, s), conv_out_shape(out2[1], p, ks, s)]

        # ---- Convolutional feature extractor (applied per time step) -------
        conv_layers = [nn.Conv2d(nchannel, filter_size, kernel_size=k_size, stride=stride, padding=padding)]
        if bn:
            conv_layers.append(nn.BatchNorm2d(filter_size))
        conv_layers.append(nn.LeakyReLU())

        conv_layers.append(nn.Conv2d(filter_size, filter_size * 2, kernel_size=k_size, stride=stride, padding=padding))
        if bn:
            conv_layers.append(nn.BatchNorm2d(filter_size * 2))
        conv_layers.append(nn.LeakyReLU())

        conv_layers.append(nn.Conv2d(filter_size * 2, filter_size * 4, kernel_size=k_size, stride=stride, padding=padding))
        if bn:
            conv_layers.append(nn.BatchNorm2d(filter_size * 4))
        conv_layers.append(nn.LeakyReLU())

        self.conv_net = nn.Sequential(*conv_layers)

        # ---- Temporal LSTM layers ------------------------------------------
        lstm_in_dim = filter_size * 4 * out3[0] * out3[1]   # flattened spatial features
        self.lstm1 = nn.LSTM(lstm_in_dim, filter_size * 4, batch_first=True)
        self.lstmbn = nn.BatchNorm1d(filter_size * 4)

        # Two parallel heads produce the H and M sequences respectively
        self.final1_lstm2 = nn.LSTM(filter_size * 4, j, batch_first=True)   # → H
        self.final2_lstm2 = nn.LSTM(filter_size * 4, j, batch_first=True)   # → M

    def forward(self, inputs1: torch.Tensor, inputs2: torch.Tensor = None):
        """
        Forward pass of the discriminator.

        Parameters
        ----------
        inputs1 : torch.Tensor, shape (N, T, C, H, W)
            Primary input sequence (real or generated data, optionally with
            the DeepX embedding concatenated along the channel dimension).
        inputs2 : torch.Tensor or None, shape (N, T, C, H, W)
            Optional secondary input sequence (used to compute the second LSTM
            head output; pass the "primed" copy for the mixed Sinkhorn loss).

        Returns
        -------
        x1 : torch.Tensor, shape (N, T, J)
            H or M output for ``inputs1``.
        x2 : torch.Tensor, shape (N, T, J)  (only when ``inputs2`` is not None)
            H or M output for ``inputs2``.
        """
        T = inputs1.shape[1]

        # Process inputs1 --------------------------------------------------
        # Merge batch and time dimensions for the conv layers
        x = inputs1.reshape(self.batch_size * T, self.nchannel, self.x_height, self.x_width)
        x = self.conv_net(x)                              # (N*T, C', H', W')
        x = x.reshape(self.batch_size, T, -1)             # (N, T, features)
        x, _ = self.lstm1(x)                              # (N, T, filter_size*4)
        if self.bn:
            x = self.lstmbn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x1, _ = self.final1_lstm2(x)                      # (N, T, J)

        if inputs2 is not None:
            # Process inputs2 for the second head
            T2 = inputs2.shape[1]
            x2 = inputs2.reshape(self.batch_size * T2, self.nchannel, self.x_height, self.x_width)
            x2 = self.conv_net(x2)
            x2 = x2.reshape(self.batch_size, T2, -1)
            x2, _ = self.lstm1(x2)
            if self.bn:
                x2 = self.lstmbn(x2.permute(0, 2, 1)).permute(0, 2, 1)
            x2, _ = self.final2_lstm2(x2)
            return x1, x2

        return x1


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class VideoDCG(nn.Module):
    """
    Spatiotemporal generator.

    Maps two Gaussian noise tensors (z, y) to a synthetic sequence of gridded
    spatial fields that mimics the real climate data distribution.

    Architecture
    ------------
    1. Noise z (temporal) and y (static) are concatenated and fed through two
       stacked LSTM layers to produce a latent temporal representation.
    2. A dense layer maps each time step's LSTM state to an 8×8 feature map.
    3. Four transposed convolution (deconv) layers upsample to the target H×W.

    The padding for each transposed convolution is computed analytically so
    that the output exactly matches the requested spatial resolution, including
    non-square domains where H ≠ W.

    Parameters
    ----------
    batch_size : int
        Mini-batch size.
    time_steps : int
        Number of time steps T in the generated sequence.
    x_h : int
        Target spatial height.
    x_w : int
        Target spatial width.
    filter_size : int
        Number of feature maps in the last deconv layer; halved in earlier layers.
    state_size : int
        Hidden size of the LSTM layers.
    nchannel : int
        Number of output channels (1 for univariate temperature).
    z_dim : int
        Dimensionality of the temporal noise z at each time step (z_dim²).
    y_dim : int
        Dimensionality of the static noise y.
    bn : bool
        Whether to use BatchNorm after each dense/deconv layer.
    output_act : str
        Output activation: ``'sigmoid'`` maps output to (0,1), ``'tanh'`` maps
        to (−1, 1), ``''`` / anything else applies no activation.
    """

    def __init__(
        self,
        batch_size: int = 8,
        time_steps: int = 32,
        x_h: int = 64,
        x_w: int = 64,
        filter_size: int = 32,
        state_size: int = 32,
        nchannel: int = 1,
        z_dim: int = 25,
        y_dim: int = 20,
        bn: bool = False,
        output_act: str = 'sigmoid',
    ):
        super().__init__()

        self.batch_size = batch_size
        self.time_steps = time_steps
        self.filter_size = filter_size
        self.state_size = state_size
        self.nchannel = nchannel
        self.n_noise_t = z_dim
        self.n_noise_y = y_dim
        self.x_height = x_h
        self.x_width = x_w
        self.bn = bn
        self.output_activation = output_act

        # ---- LSTM temporal encoder ----------------------------------------
        self.lstm1 = nn.LSTM(z_dim + y_dim, state_size, batch_first=True)
        self.lstmbn1 = nn.BatchNorm1d(state_size)
        self.lstm2 = nn.LSTM(state_size, state_size * 2, batch_first=True)
        self.lstmbn2 = nn.BatchNorm1d(state_size * 2)

        # ---- Compute deconvolution padding to hit exact target resolution --
        # We analytically solve for the padding that makes the transposed-conv
        # output match the desired shape at each of the four upsampling layers.
        def deconv_padding(h_out, h_in, k_size, stride):
            """Return (ceil) padding to achieve h_out from h_in."""
            p1 = torch.tensor(((h_in[0] - 1) * stride[0] - h_out[0] + k_size[0]) / 2)
            p2 = torch.tensor(((h_in[1] - 1) * stride[1] - h_out[1] + k_size[1]) / 2)
            return [int(abs(torch.ceil(p1))), int(abs(torch.ceil(p2)))]

        if x_h == x_w:
            # Square domain: equal strides in both spatial directions
            stride4, k_size4 = [2, 2], [6, 6]
            h4_in = [x_h // 2, x_w // 2]
            padding4 = deconv_padding([x_h, x_w], h4_in, k_size4, stride4)

            stride3, k_size3 = [2, 2], [6, 6]
            h3_in = [x_h // 4, x_w // 4]
            padding3 = deconv_padding(h4_in, h3_in, k_size3, stride3)

            stride2, k_size2 = [2, 2], [4, 4]
            h2_in = [x_h // 8, x_w // 8]
            padding2 = deconv_padding(h3_in, h2_in, k_size2, stride2)

            stride1, k_size1 = [2, 2], [2, 2]
            padding1 = deconv_padding(h2_in, [8, 8], k_size1, stride1)

        elif x_h < x_w:
            # Landscape domain: larger stride along width
            stride4, k_size4 = [2, 3], [8, 9]
            h4_in = [x_h // 2, x_w // 2]
            padding4 = deconv_padding([x_h, x_w], h4_in, k_size4, stride4)

            stride3, k_size3 = [2, 3], [6, 7]
            h3_in = [x_h // 4, x_w // 4]
            padding3 = deconv_padding(h4_in, h3_in, k_size3, stride3)

            stride2, k_size2 = [2, 3], [6, 7]
            h2_in = [x_h // 8, x_w // 8]
            padding2 = deconv_padding(h3_in, h2_in, k_size2, stride2)

            stride1, k_size1 = [2, 3], [6, 7]
            padding1 = deconv_padding(h2_in, [8, 8], k_size1, stride1)

        else:
            # Portrait domain: larger stride along height
            stride4, k_size4 = [3, 2], [9, 8]
            h4_in = [x_h // 2, x_w // 2]
            padding4 = deconv_padding([x_h, x_w], h4_in, k_size4, stride4)

            stride3, k_size3 = [3, 2], [7, 6]
            h3_in = [x_h // 4, x_w // 4]
            padding3 = deconv_padding(h4_in, h3_in, k_size3, stride3)

            stride2, k_size2 = [3, 2], [7, 6]
            h2_in = [x_h // 8, x_w // 8]
            padding2 = deconv_padding(h3_in, h2_in, k_size2, stride2)

            stride1, k_size1 = [3, 2], [7, 6]
            padding1 = deconv_padding(h2_in, [8, 8], k_size1, stride1)

        # ---- Dense layer: LSTM state → 8×8 spatial feature map ------------
        dense_layers = [nn.Linear(state_size * 2, 8 * 8 * filter_size * 4)]
        if bn:
            dense_layers.append(nn.BatchNorm1d(8 * 8 * filter_size * 4))
        dense_layers.append(nn.LeakyReLU())
        self.dense_net = nn.Sequential(*dense_layers)

        # ---- Transposed-convolution upsampling stack ----------------------
        deconv_layers = [
            nn.ConvTranspose2d(filter_size * 4, filter_size * 4, kernel_size=k_size1, stride=stride1, padding=padding1)
        ]
        if bn:
            deconv_layers.append(nn.BatchNorm2d(filter_size * 4))
        deconv_layers.append(nn.LeakyReLU())

        deconv_layers.append(
            nn.ConvTranspose2d(filter_size * 4, filter_size * 2, kernel_size=k_size2, stride=stride2, padding=padding2)
        )
        if bn:
            deconv_layers.append(nn.BatchNorm2d(filter_size * 2))
        deconv_layers.append(nn.LeakyReLU())

        deconv_layers.append(
            nn.ConvTranspose2d(filter_size * 2, filter_size, kernel_size=k_size3, stride=stride3, padding=padding3)
        )
        if bn:
            deconv_layers.append(nn.BatchNorm2d(filter_size))
        deconv_layers.append(nn.LeakyReLU())

        deconv_layers.append(
            nn.ConvTranspose2d(filter_size, nchannel, kernel_size=k_size4, stride=stride4, padding=padding4)
        )

        # Output activation maps to [0,1] (sigmoid) to match the normalised input
        if output_act == 'sigmoid':
            deconv_layers.append(nn.Sigmoid())
        elif output_act == 'tanh':
            deconv_layers.append(nn.Tanh())

        self.deconv_net = nn.Sequential(*deconv_layers)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Generate a synthetic spatiotemporal sequence from latent noise.

        Parameters
        ----------
        z : torch.Tensor, shape (N, T, z_dim)
            Temporal noise: varies at each time step to drive temporal dynamics.
        y : torch.Tensor, shape (N, y_dim)
            Static noise: constant across time steps, encoding global style.

        Returns
        -------
        torch.Tensor of shape (N, T, C, H, W)
            Generated spatiotemporal sequence, values in [0, 1] after sigmoid.
        """
        # Expand static y to match the time dimension and concatenate with z
        y_expanded = y[:, None, :].expand(self.batch_size, self.time_steps, self.n_noise_y)
        x = torch.cat([z, y_expanded], dim=-1)   # (N, T, z_dim + y_dim)

        # Temporal encoding via two stacked LSTMs
        x, _ = self.lstm1(x)   # (N, T, state_size)
        if self.bn:
            x = self.lstmbn1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x, _ = self.lstm2(x)   # (N, T, state_size*2)
        if self.bn:
            x = self.lstmbn2(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Dense layer + reshape to 8×8 spatial feature maps (per time step)
        x = x.reshape(self.batch_size * self.time_steps, -1)
        x = self.dense_net(x)                                               # (N*T, filter*4*64)
        x = x.reshape(self.batch_size * self.time_steps, self.filter_size * 4, 8, 8)

        # Transposed convolutions upsample to target (H, W)
        x = self.deconv_net(x)                                              # (N*T, C, H, W)

        # Restore batch and time dimensions
        x = x.reshape(self.batch_size, self.time_steps, self.nchannel, self.x_height, self.x_width)
        return x
