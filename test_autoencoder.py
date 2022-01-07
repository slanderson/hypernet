"""
Test saved autoencoder
"""

import sys
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pdb

from train_utils import get_data, show_model, TrainingMonitor
from train_autoencoder import (
                get_snapshot_params, 
                ROM_SIZE, 
                LR_INIT, LR_PATIENCE, COMPLETION_PATIENCE,
                MODEL_PATH
                )
from models import Scaler, Unscaler, Encoder, Decoder, Autoencoder
from config import SEED, NUM_CELLS

def main():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  # device = 'cpu'
  print(f"Using {device} device")

  if len(sys.argv) < 2:
    model_path = MODEL_PATH
  else:
    model_path = sys.argv[1]

  torch.set_default_dtype(torch.float32)
  rng = torch.Generator()
  rng = rng.manual_seed(SEED)
  np_rng = np.random.default_rng(SEED)

  mu_samples = get_snapshot_params()
  data_tuple = get_data(np_rng, mu_samples, dtype='float32')
  train_t, val_t, train_data, val_data, train_loader, val_loader = data_tuple

  scaler = Scaler(train_t).to(device)
  unscaler = Unscaler(train_t).to(device)
  enc = Encoder(NUM_CELLS, ROM_SIZE).to(device)
  dec = Decoder(NUM_CELLS, ROM_SIZE).to(device)
  auto = Autoencoder(enc, dec, scaler, unscaler).to(device)
  opt = optim.Adam(auto.parameters(), lr=LR_INIT)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1,
                                                   patience=LR_PATIENCE, verbose=True)

  monitor = TrainingMonitor(MODEL_PATH, COMPLETION_PATIENCE, auto, opt, scheduler, train=False)
  monitor.load_from_path(model_path)
  monitor.plot_training_curves()
  show_model(auto, train_data, val_data, device=device)
  plt.show()

  pdb.set_trace()

if __name__ == "__main__":
  main()
