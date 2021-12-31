"""
Train an autoencoder to give a reduced representation of the state
"""

import numpy as np
import matplotlib.pyplot as plt
import pdb

import torch
from torch import nn
import torch.optim as optim

from models import Encoder, Decoder, Autoencoder, Scaler, Unscaler
from train_utils import get_data, random_split, show_model, TrainingMonitor
from config import SEED, NUM_CELLS, TRAIN_FRAC

EPOCHS = 10
ROM_SIZE = 40
LR_INIT = 1e-3
LR_PATIENCE = 20
COMPLETION_PATIENCE = 100
MODEL_PATH = 'autoenc.pt'

MU1_LOW, MU1_HIGH = 4.25, 5.5
MU2_LOW, MU2_HIGH = 0.015, 0.03
SAMPLES_PER_MU = 3

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

def get_snapshot_params(samples_per_mu):
  mu1_samples = np.linspace(MU1_LOW, MU1_HIGH, samples_per_mu)
  mu2_samples = np.linspace(MU2_LOW, MU2_HIGH, samples_per_mu)
  mu_samples = []
  for mu1 in mu1_samples:
    for mu2 in mu2_samples:
      mu_samples += [[mu1, mu2]]
  return mu_samples

def main():

  torch.set_default_dtype(torch.float64)

  rng = torch.Generator()
  rng = rng.manual_seed(SEED)
  np_rng = np.random.default_rng(SEED)

  mu_samples = get_snapshot_params(SAMPLES_PER_MU)
  data_tuple = get_data(np_rng, mu_samples)
  train_t, val_t, train_data, val_data, train_loader, val_loader = data_tuple

  scaler = Scaler(train_t)
  unscaler = Unscaler(train_t)
  enc = Encoder(NUM_CELLS, ROM_SIZE)
  dec = Decoder(NUM_CELLS, ROM_SIZE)
  auto = Autoencoder(enc, dec, scaler, unscaler)
  loss = nn.MSELoss()
  opt = optim.Adam(auto.parameters(), lr=LR_INIT)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1,
                                                   patience=LR_PATIENCE, verbose=True)

  train_losses = []
  test_losses = []
  monitor = TrainingMonitor(MODEL_PATH, COMPLETION_PATIENCE)
  for t in range(EPOCHS):
    print("\nEpoch {}:".format(t+1))
    train_loss = train(train_loader, auto, loss, opt)
    test_loss = test(val_loader, auto, loss)
    scheduler.step(test_loss)
    test_losses += [test_loss]
    train_losses += [train_loss]
    if monitor.check_for_completion(test_loss):
      break
  print("Training complete!")

  show_model(auto, train_data, val_data, train_losses, test_losses)

  pdb.set_trace()


if __name__ == "__main__":
  main()

