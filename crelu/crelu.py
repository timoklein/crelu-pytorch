import torch
import torch.nn as nn
import torch.nn.functional as F


def crelu(input: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """CReLU activation function.

    CReLU is the concatenation of ReLU applied to x and -x: CReLU(x) = [ReLU (x), ReLU (-x)].
    Applying CReLU to a tensor of shape (N, D) results in a tensor of shape (N, 2D).

    Parameters
    ----------
    input : torch.Tensor
        Input tensor.
    dim : int
        Dimension along which the features will be concatenated. Default is 1.

    Returns
    -------
    torch.Tensor
        Tensor with CReLU applied to input.
    """

    return torch.cat((F.relu(input), F.relu(-input)), dim=dim)


class CReLU(nn.Module):
    """CReLU activation function.

    CReLU is the concatenation of ReLU applied to x and -x: CReLU(x) = [ReLU (x), ReLU (-x)].
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """Apply the CReLU function to the input tensor.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor.
        dim : int
            Dimension along which the features will be concatenated. Default is 1.

        Returns
        -------
        torch.Tensor
            Tensor with CReLU applied to input.
        """
        return crelu(input, dim)


if __name__ == "__main__":
    c = CReLU()

    # Test that it works for common liner use case
    t1 = torch.randn(32, 64)
    linear = nn.Linear(64, 64)

    assert c(t1).shape == (32, 128), "Invalid shape for CReLU linear output."

    # Test that it works for common conv use case
    t2 = torch.randn(32, 64, 16, 16)
    conv = nn.Conv2d(64, 64, 3, padding=1)

    assert c(t2).shape == (32, 128, 16, 16), "Invalid shape for CReLU conv output."
