"""
Run the burgers' equation with local PROMs
"""

import glob
import math

import numpy as np
import matplotlib.pyplot as plt

from hypernet import (
                      make_1D_grid,
                      load_or_compute_snaps,
                      POD,
                      podsize,
                      compute_local_bases,
                      inviscid_burgers_LSPG_local,
                      inviscid_burgers_LSPG,
                      compute_error,
                      plot_snaps,
                     )

import pdb


def main():

    snap_folder = 'param_snaps'
    num_clusts = 10
    energy_thresh = 0.999999
    energy_thresh_local = 0.999999
    min_size = None
    max_size = None
    overlap_frac = 0.1

    dt = 0.07
    num_steps = 500
    num_cells = 500
    xl, xu = 0, 100
    w0 = np.ones(num_cells)
    grid = make_1D_grid(xl, xu, num_cells)

    mu_samples = [
               [4.3, 0.021],
               [5.1, 0.030]
              ]
    mu_rom = [4.7, 0.026]

    # Generate or retrieve HDM snapshots
    all_snaps_list = []
    for mu in mu_samples:
        snaps = load_or_compute_snaps(mu, grid, w0, dt, num_steps, snap_folder=snap_folder)
        all_snaps_list += [snaps]

    snaps = np.hstack(all_snaps_list)   

    # construct basis using mu_samples params
    basis, sigma = POD(snaps)
    num_vecs = podsize(sigma, energy_thresh=energy_thresh)
    basis_trunc = basis[:, :num_vecs]
    local_bases, centroids = compute_local_bases(snaps, num_clusts, 
                                                 energy_thresh=energy_thresh_local, 
                                                 min_size=min_size, max_size=max_size,
                                                 overlap_frac=0.1)

    # evaluate ROM at mu_rom
    local_rom_snaps, times = inviscid_burgers_LSPG_local(grid, w0, dt, num_steps, mu_rom,
                                                  local_bases, centroids)
    rom_snaps, times = inviscid_burgers_LSPG(grid, w0, dt, num_steps, mu_rom, basis_trunc)
    hdm_snaps = load_or_compute_snaps(mu_rom, grid, w0, dt, num_steps, snap_folder=snap_folder)
    errors, rms_err = compute_error(rom_snaps, hdm_snaps)
    local_errors, local_rms_err = compute_error(local_rom_snaps, hdm_snaps)

    fig, (ax1, ax2) = plt.subplots(2)
    snaps_to_plot = range(50, 501, 50)
    plot_snaps(grid, hdm_snaps, snaps_to_plot, 
               label='HDM', fig_ax=(fig,ax1))
    plot_snaps(grid, rom_snaps, snaps_to_plot, 
               label='PROM', fig_ax=(fig,ax1), color='blue', linewidth=1)
    plot_snaps(grid, local_rom_snaps, snaps_to_plot, 
               label='Local PROM', fig_ax=(fig,ax1), color='red', linewidth=1)

    ax1.set_xlim([grid.min(), grid.max()])
    ax1.set_xlabel('x')
    ax1.set_ylabel('w')
    ax1.set_title('Comparing HDM and ROMs')
    ax1.legend()

    ax2.plot(errors, label='PROM error', color='blue')
    ax2.plot(local_errors, label='Local PROM error', color='red')
    ax2.set_xlabel('Time index')
    ax2.set_ylabel('Relative error')
    ax2.set_title('Comparing relative error')
    print('PROM rel. error:        {}'.format(rms_err))
    print('Local PROM rel. error:  {}'.format(local_rms_err))
    ax2.legend()
    plt.show()

    pdb.set_trace()


if __name__ == "__main__":
    main()
