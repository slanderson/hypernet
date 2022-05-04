"""
Train a MLP to represent the approximate residual norm
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import pdb

import torch
from torch import nn
import torch.optim as optim

from models import Hypernet
from train_utils import get_hyper_data, random_split, show_model, TrainingMonitor
from config import SEED, NUM_CELLS, TRAIN_FRAC, MU1_RANGE, MU2_RANGE, SAMPLES_PER_MU, ROM_SIZE

EPOCHS = 5000
LR_INIT = 1e-3
LR_PATIENCE = 50
COMPLETION_PATIENCE = 200
MODEL_PATH = 'hypernet.pt'

def train(loader, model, loss_fn, opt, device, verbose=False):
  size = len(loader.dataset)
  num_batches = len(loader)
  model.train()
  train_loss = 0
  for batch, (X,r) in enumerate(loader):
    X = X.to(device)
    r = r.to(device)
    out = model(X).squeeze()
    loss = loss_fn(out, r)

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

def test(loader, model, loss_fn, device):
  size = len(loader.dataset)
  num_batches = len(loader)
  model.eval()
  test_loss = 0
  with torch.no_grad():
    for X,r in loader:
      X = X.to(device)
      r = r.to(device)
      out = model(X).squeeze()
      test_loss += loss_fn(out, r).item()
  test_loss /= num_batches
  print("  Test loss: {:.7f}".format(test_loss))
  return test_loss

def main():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  # device = 'cpu'
  print(f"Using {device} device")

  torch.set_default_dtype(torch.float32)
  rng = torch.Generator()
  rng = rng.manual_seed(SEED)
  np_rng = np.random.default_rng(SEED)

  data_tuple = get_hyper_data(np_rng, dtype='float32')
  train_t, val_t, train_data, val_data, train_loader, val_loader = data_tuple

  hnet = Hypernet(2*ROM_SIZE).to(device)
  loss = nn.MSELoss()
  opt = optim.Adam(hnet.parameters(), lr=LR_INIT)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1,
                                                   patience=LR_PATIENCE, verbose=True)

  monitor = TrainingMonitor(MODEL_PATH, COMPLETION_PATIENCE, hnet, opt, scheduler)
  if len(sys.argv) > 1:
    monitor.load_from_path(sys.argv[1])
  for t in range(EPOCHS):
    print("\nEpoch {}:".format(t+1))
    train_loss = train(train_loader, hnet, loss, opt, device)
    test_loss = test(val_loader, hnet, loss, device)
    scheduler.step(test_loss)
    if monitor.check_for_completion(train_loss, test_loss):
      break
  print("Training complete!")

  monitor.plot_training_curves()
  plt.show()

  pdb.set_trace()


if __name__ == "__main__":
  main()

