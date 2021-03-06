"""
Pytorch neural network models
"""

from torch import nn
import pdb

class Encoder(nn.Module):
  def __init__(self, hdm_size, rom_size, carlberg=True):
    super(Encoder, self).__init__()
    if carlberg:
      self.elu_stack = nn.Sequential(
          nn.Conv1d(1, 8, 25, stride=2, padding=12),
          nn.ELU(),
          nn.Conv1d(8, 16, 25, stride=4, padding=12),
          nn.ELU(),
          nn.Conv1d(16, 32, 25, stride=4, padding=12),
          nn.ELU(),
          nn.Conv1d(32, 64, 25, stride=4, padding=12),
          nn.ELU(),
          nn.Flatten(),
          nn.Linear(64*4, rom_size),
          )
    else:
      self.elu_stack = nn.Sequential(
          nn.Conv1d(1, 8, 4, stride=2, padding=1),
          nn.ELU(),
          nn.Conv1d(8, 8, 5, stride=1, padding=2),
          nn.ELU(),
          nn.Conv1d(8, 32, 4, stride=4, padding=0),
          nn.ELU(),
          nn.Conv1d(32, 32, 5, stride=1, padding=2),
          nn.ELU(),
          nn.Conv1d(32, 128, 4, stride=4, padding=0),
          nn.ELU(),
          nn.Conv1d(128, 128, 5, stride=1, padding=2),
          nn.ELU(),
          nn.Flatten(),
          nn.Linear(128*16, rom_size),
          )

  def forward(self, x):
    z = self.elu_stack(x)
    return z


class Decoder(nn.Module):
  def __init__(self, hdm_size, rom_size, carlberg=True):
    super(Decoder, self).__init__()
    if carlberg:
      self.elu_stack = nn.Sequential(
          nn.Linear(rom_size, 64*4),
          nn.ELU(),
          nn.Unflatten(1, (64, 4)),
          nn.ConvTranspose1d(64, 64, 25, stride=4, padding=12, output_padding=3),
          nn.ELU(),
          nn.ConvTranspose1d(64, 32, 25, stride=4, padding=12, output_padding=3),
          nn.ELU(),
          nn.ConvTranspose1d(32, 16, 25, stride=4, padding=12, output_padding=3),
          nn.ELU(),
          nn.ConvTranspose1d(16, 1, 25, stride=2, padding=12, output_padding=1),
          )
    else:
      self.elu_stack = nn.Sequential(
          nn.Linear(rom_size, 128*16),
          nn.ELU(),
          nn.Unflatten(1, (128, 16)),
          nn.Conv1d(128, 128, 5, stride=1, padding=2),
          nn.ELU(),
          nn.ConvTranspose1d(128, 32, 4, stride=4, padding=0),
          nn.ELU(),
          nn.Conv1d(32, 32, 5, stride=1, padding=2),
          nn.ELU(),
          nn.ConvTranspose1d(32, 8, 4, stride=4, padding=0),
          nn.ELU(),
          nn.Conv1d(8, 8, 5, stride=1, padding=2),
          nn.ELU(),
          nn.ConvTranspose1d(8, 1, 4, stride=2, padding=1),
          )

  def forward(self, z):
    x = self.elu_stack(z)
    return x


class Autoencoder(nn.Module):
  def __init__(self, enc, dec, scaler, unscaler):
    super(Autoencoder, self).__init__()
    self.scaler = scaler
    self.unscaler = unscaler
    self.enc = enc
    self.dec = dec

  def forward(self, x):
    xs = self.scaler(x)
    z = self.enc(xs)
    xh = self.dec(z)
    xhu = self.unscaler(xh)
    return xhu


class Scaler(nn.Module):
  def __init__(self, train):
    super(Scaler, self).__init__()
    self.min = train.min()
    self.max = train.max()

  def forward(self, x):
    out = (x - self.min) / (self.max - self.min)
    return out


class Unscaler(nn.Module):
  def __init__(self, train):
    super(Unscaler, self).__init__()
    self.min = train.min()
    self.max = train.max()

  def forward(self, x):
    out = self.min + x * (self.max - self.min)
    return out


class Hypernet(nn.Module):
  def __init__(self, in_size):
    super(Hypernet, self).__init__()
    self.in_size = in_size
    self.elu_stack = nn.Sequential(
        nn.Linear(in_size, in_size),
        nn.ELU(),
        nn.Linear(in_size, in_size),
        nn.ELU(),
        nn.Linear(in_size, in_size),
        nn.ELU(),
        nn.Linear(in_size, in_size),
        nn.ELU(),
        nn.Linear(in_size, in_size),
        nn.ELU(),
        nn.Linear(in_size, 1),
        )

  def forward(self, x):
    r = self.elu_stack(x)
    return r
