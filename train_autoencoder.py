"""
Train an autoencoder to give a reduced representation of the state
"""

import numpy as np
import matplotlib.pyplot as plt
import pdb

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

from hypernet import (
                      make_1D_grid,
                      load_or_compute_snaps
                     )

SEED = 1234557
EPOCHS = 500
TRAIN_FRAC = 0.8
BATCH_SIZE = 40
LR = 1e-3
HDM_SIZE = 512
ROM_SIZE = 40

class Encoder(nn.Module):
  def __init__(self, hdm_size, rom_size):
    super(Encoder, self).__init__()
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

  def forward(self, x):
    z = self.elu_stack(x)
    return z

class Decoder(nn.Module):
  def __init__(self, hdm_size, rom_size):
    super(Decoder, self).__init__()
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

def load_data():
  snap_folder = 'param_snaps'
  samples_per_mu = 3

  dt = 0.07
  num_steps = 500
  num_cells = 512
  xl, xu = 0, 100
  w0 = np.ones(num_cells)
  grid = make_1D_grid(xl, xu, num_cells)

  mu1_samples = np.linspace(4.25, 5.5, samples_per_mu)
  mu2_samples = np.linspace(0.015, 0.03, samples_per_mu)
  mu_samples = []
  for mu1 in mu1_samples:
    for mu2 in mu2_samples:
      mu_samples += [[mu1, mu2]]

  # Generate or retrive HDM snapshots
  all_snaps_list = []
  for mu in mu_samples:
      snaps = load_or_compute_snaps(mu, grid, w0, dt, num_steps, snap_folder=snap_folder)
      all_snaps_list += [snaps[:, :-1]]

  snaps = np.hstack(all_snaps_list)   
  ref = snaps[:, 0]
  snaps -= ref[:, np.newaxis]
  return snaps.T, ref

def random_split(snaps, frac, rng):
  n = snaps.shape[0]
  num1 = int(n*frac)
  perm = rng.permutation(n)
  snaps_perm = snaps[perm, :]
  snaps1 = snaps_perm[:num1, :]
  snaps2 = snaps_perm[num1:, :]
  return snaps1, snaps2

def train(loader, model, loss_fn, opt, verbose=False):
  size = len(loader.dataset)
  num_batches = len(loader)
  model.train()
  train_loss = 0
  for batch, (X,) in enumerate(loader):
    out = model(X)
    loss = loss_fn(out, X)

    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.no_grad():
      train_loss += loss.item()

    if verbose:
      if batch % 20 == 0:
        loss, current = loss.item(), batch * len(X)
        print(  "loss: {:.7f}  [{:5d} / {:5d}]".format(loss, current, size))
  train_loss /= num_batches
  print("  Train loss: {:.7f}".format(train_loss))
  return train_loss

def test(loader, model, loss_fn):
  size = len(loader.dataset)
  num_batches = len(loader)
  model.eval()
  test_loss = 0
  with torch.no_grad():
    for X, in loader:
      out = model(X)
      test_loss += loss_fn(out, X).item()
  test_loss /= num_batches
  print("  Test loss: {:.7f}".format(test_loss))
  return test_loss

def show_model(model, train, test):
  fig, (ax1, ax2) = plt.subplots(2, 3)
  with torch.no_grad():
    for i, ax in enumerate(ax1):
      x = train[i][0].unsqueeze(0)
      out = model(x)
      ax.plot(x.squeeze(), linewidth=2, color='k')
      ax.plot(out.squeeze(), linewidth=1, color='r')
    for i, ax in enumerate(ax2):
      x = test[i][0].unsqueeze(0)
      out = model(x)
      ax.plot(x.squeeze(), linewidth=2, color='k')
      ax.plot(out.squeeze(), linewidth=1, color='r')
  
  plt.show()

def main():

  torch.set_default_dtype(torch.float64)

  rng = torch.Generator()
  rng = rng.manual_seed(SEED)
  np_rng = np.random.default_rng(SEED)

  snaps_np, ref = load_data()
  train_np, val_np = random_split(snaps_np, TRAIN_FRAC, np_rng)
  train_t = torch.from_numpy(train_np).unsqueeze(1)
  val_t = torch.from_numpy(val_np).unsqueeze(1)
  train_data = TensorDataset(train_t)
  val_data = TensorDataset(val_t)
  train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
  val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
  
  scaler = Scaler(train_t)
  unscaler = Unscaler(train_t)
  enc = Encoder(HDM_SIZE, ROM_SIZE)
  dec = Decoder(HDM_SIZE, ROM_SIZE)
  auto = Autoencoder(enc, dec, scaler, unscaler)
  loss = nn.MSELoss()
  opt = optim.Adam(auto.parameters(), lr=LR)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1,
                                                   patience=20, verbose=True)

  train_losses = []
  test_losses = []
  for t in range(EPOCHS):
    print("\nEpoch {}:".format(t+1))
    train_loss = train(train_loader, auto, loss, opt)
    test_loss = test(val_loader, auto, loss)
    scheduler.step(test_loss)
    test_losses += [test_loss]
    train_losses += [train_loss]
  print("Training complete!")

  fig, ax = plt.subplots()
  ax.semilogy(test_losses, label='Test loss')
  ax.semilogy(train_losses, label='Train loss')
  ax.set_xlabel('Epoch')
  ax.set_ylabel('MSE loss')
  ax.set_title('Training curves')
  ax.legend()

  show_model(auto, train_data, val_data)

  pdb.set_trace()


if __name__ == "__main__":
  main()

