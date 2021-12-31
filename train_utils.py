"""
Basic utility functions/classes for training the autoencoder
"""

import pdb
import numpy as np
import matplotlib.pyplot as plt
import torch
from hypernet import (
                      make_1D_grid,
                      load_or_compute_snaps
                     )
from torch.utils.data import DataLoader, TensorDataset
from config import (
                    TRAIN_FRAC, BATCH_SIZE,
                    DT, NUM_STEPS, NUM_CELLS, W0, GRID, SNAP_FOLDER
                   )

class TrainingMonitor:
  def __init__(self, model_path, patience):
    self.model_path = model_path
    self.best_crit = 1E16
    self.patience = patience
    self.its_since_improvement = 0

  def check_for_completion(self, test_crit):
    self.its_since_improvement += 1
    if test_crit < self.best_crit:
      self.best_crit = test_crit
      self.its_since_improvement = 0
    print('  Its since improvement: {}'.format(self.its_since_improvement))
    if self.its_since_improvement > self.patience:
      return True

    return False

def random_split(snaps, frac, rng):
  n = snaps.shape[0]
  num1 = int(n*frac)
  perm = rng.permutation(n)
  snaps_perm = snaps[perm, :]
  snaps1 = snaps_perm[:num1, :]
  snaps2 = snaps_perm[num1:, :]
  return snaps1, snaps2

def get_data(np_rng, mu_samples):
  snaps_np, ref = load_data(mu_samples)
  train_np, val_np = random_split(snaps_np, TRAIN_FRAC, np_rng)
  train_t = torch.from_numpy(train_np).unsqueeze(1)
  val_t = torch.from_numpy(val_np).unsqueeze(1)
  train_data = TensorDataset(train_t)
  val_data = TensorDataset(val_t)
  train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
  val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
  return train_t, val_t, train_data, val_data, train_loader, val_loader

def load_data(mu_samples):
  # Generate or retrive HDM snapshots
  all_snaps_list = []
  for mu in mu_samples:
      snaps = load_or_compute_snaps(mu, GRID, W0, DT, NUM_STEPS, snap_folder=SNAP_FOLDER)
      all_snaps_list += [snaps[:, :-1]]

  snaps = np.hstack(all_snaps_list)   
  ref = snaps[:, 0]
  snaps -= ref[:, np.newaxis]
  return snaps.T, ref

def show_model(model, train, test, train_losses, test_losses):
  fig_curves, ax_curves = plt.subplots()
  ax_curves.semilogy(test_losses, label='Test loss')
  ax_curves.semilogy(train_losses, label='Train loss')
  ax_curves.set_xlabel('Epoch')
  ax_curves.set_ylabel('MSE loss')
  ax_curves.set_title('Training curves')
  ax_curves.legend()

  fig, (ax1, ax2) = plt.subplots(2, 3)
  with torch.no_grad():
    for i, ax in enumerate(ax1):
      x = train[i][0].unsqueeze(0)
      out = model(x)
      ax.plot(x.squeeze(), linewidth=2, color='k')
      ax.plot(out.squeeze(), linewidth=1, color='r')
      ax.set_title('Training sample')
    for i, ax in enumerate(ax2):
      x = test[i][0].unsqueeze(0)
      out = model(x)
      ax.plot(x.squeeze(), linewidth=2, color='k')
      ax.plot(out.squeeze(), linewidth=1, color='r')
      ax.set_title('Test sample')
  fig.tight_layout()
  
  plt.show()
