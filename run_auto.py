"""
Run the autoencoder PROM, and compare it to the HDM at an out-of-sample
point
"""

import glob
import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch

from hypernet import (load_or_compute_snaps, make_1D_grid, inviscid_burgers_LSPG,
                      inviscid_burgers_man, plot_snaps, POD)
from models import Scaler, Unscaler, Encoder, Decoder, Autoencoder
from train_utils import TrainingMonitor
from test_autoencoder import load_autoencoder_monitor
from config import DT, NUM_STEPS, NUM_CELLS, GRID, W0

def load_autoencoder(model_path, device='cpu'):
  monitor, ref = load_autoencoder_monitor(model_path, device, plot=False)
  return monitor.model, ref

def project_onto_autoencoder(snaps, auto, ref):
  snaps_deref = snaps - np.expand_dims(ref, 1)
  snaps_t = torch.Tensor(snaps_deref.T)
  with torch.no_grad():
    out = auto(snaps_t.unsqueeze(1))
  out_np = out.squeeze().numpy().T
  out_ref = out_np + np.expand_dims(ref, 1)
  return out_ref

def compare_snaps(snaps_to_plot, inds_to_plot, labels, colors, linewidths):
  fig, ax = plt.subplots()
  for i, snaps in enumerate(snaps_to_plot):
    plot_snaps(GRID, snaps, inds_to_plot, 
               label=labels[i], 
               fig_ax=(fig, ax), 
               color=colors[i],
               linewidth=linewidths[i])

  ax.set_xlim([GRID.min(), GRID.max()])
  ax.set_xlabel('x')
  ax.set_ylabel('w')
  ax.set_title('Comparing HDM and ROM')
  ax.legend()
  plt.show()

def main():

    model_path = 'autoenc_5k_1819ep.pt'
    snap_folder = 'param_snaps'
    num_vecs = 50

    mu_samples = [
               [4.3, 0.021],
               [5.1, 0.030]
              ]
    mu_rom = [4.875, 0.0225]

    # Generate or retrive HDM snapshots
    all_snaps_list = []
    for mu in mu_samples:
        snaps = load_or_compute_snaps(mu, GRID, W0, DT, NUM_STEPS, snap_folder=snap_folder)
        all_snaps_list += [snaps]

    snaps = np.hstack(all_snaps_list)   

    # construct basis using mu_samples params
    basis, sigma = POD(snaps)
    basis_trunc = basis[:, :num_vecs]

    # load autoencoder
    auto, ref = load_autoencoder(model_path, device='cpu')

    # evaluate ROM at mu_rom
    # rom_snaps, times = inviscid_burgers_LSPG(GRID, W0, DT, NUM_STEPS, mu_rom, basis_trunc)
    hdm_snaps = load_or_compute_snaps(mu_rom, GRID, W0, DT, NUM_STEPS, snap_folder=snap_folder)
    proj_snaps = project_onto_autoencoder(hdm_snaps, auto, ref)
    man_snaps, man_times = inviscid_burgers_man(GRID, W0, DT, NUM_STEPS, mu_rom, auto, ref)
    man_its, man_jac, man_res, man_ls = man_times

    inds_to_plot = range(0, NUM_STEPS, NUM_STEPS//5)
    snaps_to_plot = [hdm_snaps, proj_snaps, man_snaps]
    labels = ['HDM', 'Manifold Projections', 'Manifold PROM']
    colors = ['black', 'red', 'green']
    linewidths = [2, 1, 1]
    compare_snaps(snaps_to_plot, inds_to_plot, labels, colors, linewidths)


    pdb.set_trace()


if __name__ == "__main__":
    main()
