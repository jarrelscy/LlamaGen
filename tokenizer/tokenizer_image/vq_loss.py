# Modified from:
#   taming-transformers:  https://github.com/CompVis/taming-transformers
#   muse-maskgit-pytorch: https://github.com/lucidrains/muse-maskgit-pytorch/blob/main/muse_maskgit_pytorch/vqgan_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Tuple, Union
import warnings


from tokenizer.tokenizer_image.lpips import LPIPS
from tokenizer.tokenizer_image.discriminator_patchgan import NLayerDiscriminator as PatchGANDiscriminator
from tokenizer.tokenizer_image.discriminator_stylegan import Discriminator as StyleGANDiscriminator

def _fspecial_gauss_1d(size: int, sigma: float) -> Tensor:
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input: Tensor, win: Tensor) -> Tensor:
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def _ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float,
    win: Tensor,
    size_average: bool = True,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03)
) -> Tuple[Tensor, Tensor]:
    r""" Calculate ssim index for X and Y

    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        data_range (float or int): value range of input images. (usually 1.0 or 255)
        win (torch.Tensor): 1-D gauss kernel
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map    
    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs, ssim_map, cs_map


def ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    nonnegative_ssim: bool = True,
) -> Tensor:
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu

    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    #if not X.type() == Y.type():
    #    raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, cs, ssim_map, cs_map = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean(), ssim_map, cs_map
    else:
        return ssim_per_channel.mean(1), ssim_map, cs_map


class SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range: float = 255,
        size_average: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        channel: int = 3,
        spatial_dims: int = 2,
        K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
        nonnegative_ssim: bool = True,
    ) -> None:
        r""" class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        return ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
        )


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.softplus(-logits_real))
    loss_fake = torch.mean(F.softplus(logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def non_saturating_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(logits_real),  logits_real))
    loss_fake = torch.mean(F.binary_cross_entropy_with_logits(torch.zeros_like(logits_fake), logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def hinge_gen_loss(logit_fake):
    return -torch.mean(logit_fake)


def non_saturating_gen_loss(logit_fake):
    return torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(logit_fake),  logit_fake))


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


# --- Helper Function for Feature Extraction (similar to explore.ipynb) ---
def get_intermediate_features(discriminator, layer_index, x):
    """Extracts features from an intermediate layer of the NLayerDiscriminator."""
    # Assumes discriminator.main holds the sequential layers
    if not hasattr(discriminator, 'main') or not isinstance(discriminator.main, nn.Sequential):
        raise TypeError("Discriminator structure not as expected (missing .main or not Sequential)")

    all_layers = list(discriminator.main.children())
    num_layers = len(all_layers)

    if layer_index >= num_layers:
        # Try to find a sensible layer if index is out of bounds (e.g., last layer before final conv)
        # Assuming final layer is Conv2d, find the last non-Conv2d layer index
        conv_indices = [i for i, layer in enumerate(all_layers) if isinstance(layer, nn.Conv2d)]
        if len(conv_indices) > 1:
             # Use the layer before the last Conv2d
             fallback_index = conv_indices[-1] -1
             print(f"Warning: Requested layer index {layer_index} >= {num_layers}. Falling back to layer {fallback_index}.")
             layer_index = fallback_index
        else: # Cannot determine fallback, use the second to last layer overall
             fallback_index = num_layers - 2
             print(f"Warning: Requested layer index {layer_index} >= {num_layers}. Falling back to layer {fallback_index}.")
             layer_index = max(0, fallback_index) # Ensure it's not negative

    # Create a sequential model up to the desired layer
    feature_extractor = nn.Sequential(*all_layers[:layer_index+1])

    # Ensure the extractor is in eval mode if the main discriminator is
    if not discriminator.training:
        feature_extractor.eval()
    feature_extractor.to(x.device)

    with torch.no_grad(): # No gradients needed for feature extraction itself
        features = feature_extractor(x)
    return features
# --- End Helper Function ---

class VQLoss(nn.Module):
    def __init__(self, disc_start, disc_loss="hinge", disc_dim=64, disc_type='patchgan', image_size=256,
                 disc_num_layers=3, disc_in_channels=3, disc_weight=1.0, disc_adaptive_weight = False,
                 gen_adv_loss='hinge', reconstruction_loss='l2', reconstruction_weight=1.0,
                 codebook_weight=1.0, perceptual_weight=1.0,
                 disc_feature_weight=1.0, disc_feature_layer=7, # <-- Add new parameters
                 disc_patchgan_actnorm=False, # Add flag for PatchGAN ActNorm
                 ssim_weight=0.0, ssim_win_size=11, ssim_win_sigma=1.5,
    ):
        super().__init__()
        # discriminator loss
        assert disc_type in ["patchgan", "stylegan"]
        assert disc_loss in ["hinge", "vanilla", "non-saturating"]
        if disc_type == "patchgan":
            self.discriminator = PatchGANDiscriminator(
                input_nc=disc_in_channels, 
                n_layers=disc_num_layers,
                ndf=disc_dim,
                use_actnorm=disc_patchgan_actnorm,
            )
        elif disc_type == "stylegan":
            self.discriminator = StyleGANDiscriminator(
                input_nc=disc_in_channels, 
                image_size=image_size,
            )
        else:
            raise ValueError(f"Unknown GAN discriminator type '{disc_type}'.")
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        elif disc_loss == "non-saturating":
            self.disc_loss = non_saturating_d_loss
        else:
            raise ValueError(f"Unknown GAN discriminator loss '{disc_loss}'.")
        self.discriminator_iter_start = disc_start
        self.disc_weight = disc_weight
        self.disc_adaptive_weight = disc_adaptive_weight
        self.disc_feature_weight = disc_feature_weight
        self.disc_feature_layer = disc_feature_layer

        assert gen_adv_loss in ["hinge", "non-saturating"]
        # gen_adv_loss
        if gen_adv_loss == "hinge":
            self.gen_adv_loss = hinge_gen_loss
        elif gen_adv_loss == "non-saturating":
            self.gen_adv_loss = non_saturating_gen_loss
        else:
            raise ValueError(f"Unknown GAN generator loss '{gen_adv_loss}'.")

        # perceptual loss
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        # reconstruction loss
        if reconstruction_loss == "l1":
            self.rec_loss = F.l1_loss
        elif reconstruction_loss == "l2":
            self.rec_loss = F.mse_loss
        else:
            raise ValueError(f"Unknown rec loss '{reconstruction_loss}'.")
        self.rec_weight = reconstruction_weight

        # ssim loss
        self.ssim_weight = ssim_weight
        if self.ssim_weight > 0:
            self.ssim_loss = SSIM(data_range=1.0, # Assuming inputs are in [-1, 1], so range is 2.0? Or normalize to [0,1]? Let's assume data_range=1.0 for normalized [0,1] inputs? The LPIPS uses normalized [-1,1]. SSIM usually expects [0, 1] or [0, 255]. Let's transform inputs to [0,1] for SSIM.
                                  size_average=True, 
                                  win_size=ssim_win_size, 
                                  win_sigma=ssim_win_sigma, 
                                  channel=1) 

        # codebook loss
        self.codebook_weight = codebook_weight

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight.detach()

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx, global_step, last_layer=None, 
                logger=None, log_every=100):
        # generator update
        if optimizer_idx == 0:
            # reconstruction loss
            rec_loss = self.rec_loss(inputs.contiguous(), reconstructions.contiguous())

            # perceptual loss
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            p_loss = torch.mean(p_loss)

            # ssim loss (inputs shifted to [0, 1])
            ssim_val = torch.tensor(0.0, device=inputs.device)
            if self.ssim_weight > 0:
                # Shift inputs from [-1, 1] to [0, 1] for SSIM calculation
                inputs_0_1 = (inputs.contiguous() + 1.0) / 2.0
                reconstructions_0_1 = (reconstructions.contiguous() + 1.0) / 2.0
                ssim_val, _, _ = self.ssim_loss(inputs_0_1.mean(dim=1, keepdim=True), reconstructions_0_1.mean(dim=1, keepdim=True))

            # SSIM loss term (1 - SSIM value)
            ssim_loss_term = 1.0 - ssim_val

            # discriminator loss
            logits_fake = self.discriminator(reconstructions.contiguous())
            generator_adv_loss = self.gen_adv_loss(logits_fake)
            
            # discriminator feature loss
            disc_feature_loss = torch.tensor(0.0, device=inputs.device)
            if self.disc_feature_weight > 0 and isinstance(self.discriminator, PatchGANDiscriminator): # Only for PatchGAN for now
                # Ensure discriminator is on the same device and in eval mode for feature extraction consistency
                # (though gradients won't flow back through it here)
                self.discriminator.eval()
                feat_orig = get_intermediate_features(self.discriminator, self.disc_feature_layer, inputs.contiguous())
                feat_recon = get_intermediate_features(self.discriminator, self.disc_feature_layer, reconstructions.contiguous())
                self.discriminator.train() # Return to train mode if it was training

                # Calculate MSE loss averaged over all dimensions (like explore.ipynb)
                disc_feature_loss = F.mse_loss(feat_orig, feat_recon) # reduction='mean' by default

            if self.disc_adaptive_weight:
                null_loss = self.rec_weight * rec_loss + self.perceptual_weight * p_loss
                disc_adaptive_weight = self.calculate_adaptive_weight(null_loss, generator_adv_loss, last_layer=last_layer)
            else:
                disc_adaptive_weight = 1
            disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start)
            
            # Combine all losses
            loss = self.rec_weight * rec_loss + \
                self.perceptual_weight * p_loss + \
                self.ssim_weight * ssim_loss_term + \
                disc_adaptive_weight * disc_weight * generator_adv_loss + \
                self.disc_feature_weight * disc_feature_loss + \
                codebook_loss[0] + codebook_loss[1] + codebook_loss[2]
            
            # Prepare components for logging
            log_dict = {
                "rec_loss": (self.rec_weight * rec_loss).detach(),
                "perceptual_loss": (self.perceptual_weight * p_loss).detach(),
                "ssim_loss": (self.ssim_weight * ssim_loss_term).detach(),
                "gen_adv_loss": (disc_adaptive_weight * disc_weight * generator_adv_loss).detach(),
                "disc_feature_loss": (self.disc_feature_weight * disc_feature_loss).detach(),
                "vq_loss": codebook_loss[0].detach(),
                "commit_loss": codebook_loss[1].detach(),
                "entropy_loss": codebook_loss[2].detach(),
                "codebook_usage": codebook_loss[3]
            }
            if global_step % log_every == 0:
                # Additionally print to console
                logger.info(f"(Generator) rec_loss: {log_dict['rec_loss']:.4f}, perceptual_loss: {log_dict['perceptual_loss']:.4f}, "
                            f"ssim_loss: {log_dict['ssim_loss']:.4f}, vq_loss: {log_dict['vq_loss']:.4f}, "
                            f"commit_loss: {log_dict['commit_loss']:.4f}, entropy_loss: {log_dict['entropy_loss']:.4f}, "
                            f"codebook_usage: {log_dict['codebook_usage']:.4f}, "
                            f"gen_adv_loss: {log_dict['gen_adv_loss']:.4f}, "
                            f"disc_feature_loss: {log_dict['disc_feature_loss']:.4f}")
            return loss, log_dict

        # discriminator update
        if optimizer_idx == 1:
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())

            disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start)
            d_adversarial_loss = disc_weight * self.disc_loss(logits_real, logits_fake)
            
            log_dict = {
                "disc_loss": d_adversarial_loss.detach(),
                "logits_real": logits_real.detach().mean(),
                "logits_fake": logits_fake.detach().mean()
            }
            if global_step % log_every == 0:
                logger.info(f"(Discriminator) discriminator_adv_loss: {log_dict['disc_loss']:.4f}, "
                            f"disc_weight: {disc_weight:.4f}, logits_real: {log_dict['logits_real']:.4f}, "
                            f"logits_fake: {log_dict['logits_fake']:.4f}")
            return d_adversarial_loss, log_dict