import torch
from torch import nn
from torch.nn import functional as f


class SmoothedAdaptiveNormalization(nn.Module):

    def __init__(self, pad: int = 10):
        super(SmoothedAdaptiveNormalization, self).__init__()
        self._pad = pad

    def forward(self, image: torch.Tensor):
        padded = F.pad(image, (2 * self._pad, 2 * self._pad, 2 * self._pad, 2 * self._pad), mode='replicate')
        kernel_size = self._pad * 2 + 1
        min_pooled = nn.MaxPool2d(kernel_size=kernel_size, stride=1)(-padded) * -1
        max_pooled = nn.MaxPool2d(kernel_size=kernel_size, stride=1)(padded)
        pool_diff = max_pooled - min_pooled
        min_unfold = f.unfold(min_pooled, kernel_size=kernel_size).view(*image.shape[:2], -1, *image.shape[2:])
        diff_unfold = f.unfold(pool_diff, kernel_size=kernel_size).view(*image.shape[:2], -1, *image.shape[2:])
        normed = image.unsqueeze(2) - min_unfold
        normed /= diff_unfold + 1e-8
        return normed.mean(dim=2)
