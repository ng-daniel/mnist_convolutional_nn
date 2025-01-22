import torch.nn as nn

"""
File containing the model class definition and model architecture.
"""

class VGG(nn.Module):
  """
  A scaled down version of the VGG convolutional neural netwoork with
  less convolution layers due to small initial image size (28x28).

  Takes input shape (N, 1, 28, 28) where N is the batch size.

  Outputs shape (N, 10) for 10 classes / possible digits.
  """
  def __init__(self):
    super().__init__()
    self.conv_block_1 = nn.Sequential(
        # N x 1 x 28 x 28
        nn.Conv2d(1, 16, stride=1, padding=1, kernel_size=3),
        nn.ReLU(),
        # N x 16 x 28 x 28
        nn.Conv2d(16, 16, stride=1, padding=1, kernel_size=3),
        nn.ReLU(),
        # N x 16 x 28 x 28
        nn.Conv2d(16, 16, stride=1, padding=1, kernel_size=3),
        nn.ReLU(),
        # N x 16 x 28 x 28
        nn.MaxPool2d(kernel_size=2)
        # N x 16 x 14 x 14
    )
    self.conv_block_2 = nn.Sequential(
        # N x 16 x 14 x 14
        nn.Conv2d(16, 16, stride=1, padding=1, kernel_size=3),
        nn.ReLU(),
        # N x 16 x 14 x 14
        nn.Conv2d(16, 16, stride=1, padding=1, kernel_size=3),
        nn.ReLU(),
        # N x 16 x 14 x 14
        nn.Conv2d(16, 16, stride=1, padding=1, kernel_size=3),
        nn.ReLU(),
        # N x 16 x 14 x 14
        nn.MaxPool2d(kernel_size=2)
        # N x 16 x 7 x 7
    )
    self.fully_connected_linear = nn.Sequential(
        # N x 16 x 7 x 7
        nn.Flatten(),
        # N x 784
        nn.Linear(784, 10)
        # N x 10
    )

  def forward(self, x):
    """
    Forward propagation through the model's layers.
    """
    # N x 1 x 28 x 28
    x = self.conv_block_1(x)
    # N x 16 x 14 x 14
    x = self.conv_block_2(x)
    # N x 16 x 7 x 7
    x = self.fully_connected_linear(x)
    # N x 10
    return x