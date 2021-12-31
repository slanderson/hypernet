"""
Build a parameterized ROM with a global ROB, and compare it to the HDM at an out-of-sample
point
"""

import glob
import pdb

import numpy as np
import matplotlib.pyplot as plt

from hypernet import (load_or_compute_snaps, make_1D_grid, inviscid_burgers_LSPG,
                      plot_snaps, POD)


def main():

    snap_folder = 'param_snaps'
    num_vecs = 50

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

    # Generate or retrive HDM snapshots
    all_snaps_list = []
    for mu in mu_samples:
        snaps = load_or_compute_snaps(mu, grid, w0, dt, num_steps, snap_folder=snap_folder)
        all_snaps_list += [snaps]

    snaps = np.hstack(all_snaps_list)   

    # construct basis using mu_samples params
    basis, sigma = POD(snaps)
    basis_trunc = basis[:, :num_vecs]

    # evaluate ROM at mu_rom
    rom_snaps, times = inviscid_burgers_LSPG(grid, w0, dt, num_steps, mu_rom, basis_trunc)
    hdm_snaps = load_or_compute_snaps(mu_rom, grid, w0, dt, num_steps, snap_folder=snap_folder)

    fig, ax = plt.subplots()
    snaps_to_plot = range(50, 501, 50)
    plot_snaps(grid, hdm_snaps, snaps_to_plot, 
               label='HDM', fig_ax=(fig,ax))
    plot_snaps(grid, rom_snaps, snaps_to_plot, 
               label='PROM', fig_ax=(fig,ax), color='blue', linewidth=1)

    ax.set_xlim([grid.min(), grid.max()])
    ax.set_xlabel('x')
    ax.set_ylabel('w')
    ax.set_title('Comparing HDM and ROM')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
