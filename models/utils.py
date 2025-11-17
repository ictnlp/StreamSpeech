import torch
import torch.nn as nn


# Empirically, 0.1 worked best in GAN-based vocoders for audio stability
# The small negative slope (0.1) allows a small gradient to flow even for negative activations
# Helpful in preventing "dead" neurons
LEAKY_RELU_SLOPE = 0.1


def init_weights(module: nn.Module, mean: float = 0.0, std: float = 0.01):
    """
    Initializes convolutional weights with values from the normal distribution (mean = 0, std = 0.01)

    If this is not overridden, conv layers use Kaiming uniform initialization by default.
    
    HiFi-GAN overrides this because
    - GANs are very sensitive to initialization
    - The authors found that small random normal weights stabilize early training and reduce artifacts
    """

    classname = module.__class__.__name__

    if classname.find("Conv") != -1:
        module.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Compute symmetric padding for 1D convs."""
    return int((kernel_size * dilation - dilation) // 2)
