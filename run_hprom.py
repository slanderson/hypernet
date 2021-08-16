"""
Run an HPROM for the 1D burgers' equation
"""

import glob

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.optimize as opt

from lsqnonneg import lsqnonneg
from hypernet import (
                      make_1D_grid,
                      load_or_compute_snaps,
                      POD,
                      compute_ECSW_training_matrix,
                      inviscid_burgers_ecsw,
                      inviscid_burgers_res,
                      inviscid_burgers_jac,
                      plot_snaps
                     )

import pdb


def main():

    snap_folder = 'param_snaps'
    num_vecs = 50
    ecsw_max_support = 250
    ecsw_err_thresh = 1E-16
    snap_sample_factor = 10

    dt = 0.07
    num_steps = 400
    num_cells = 500
    xl, xu = 0, 100
    w0 = np.ones(num_cells)
    grid = make_1D_grid(xl, xu, num_cells)

    mu_list = [
               [4.7, 0.026],
              ]
    mu_rom = [4.7, 0.026]

    # Generate or retrive HDM snapshots
    all_snaps_list = []
    for mu in mu_list:
        snaps = load_or_compute_snaps(mu, grid, w0, dt, num_steps, snap_folder=snap_folder)
        all_snaps_list += [snaps[:, :num_steps]]

    all_snaps = np.hstack(all_snaps_list)   

    # construct basis using mu_list params
    basis, sigma = POD(all_snaps)
    basis_trunc = basis[:, :num_vecs]

    # Perform ECSW hyper-reduction
    Clist = []
    for imu, mu in enumerate(mu_list):
        mu_snaps = all_snaps_list[imu]
        Ci = compute_ECSW_training_matrix(mu_snaps[:, 1:num_steps], 
                                          mu_snaps[:, 0:num_steps-1], 
                                          basis_trunc, inviscid_burgers_res,
                                          inviscid_burgers_jac, grid, dt, mu)
        Clist += [Ci]
    C = np.vstack(Clist)

    weights1, rnormsq1, res1 = lsqnonneg(C[:,5:], C[:,5:].sum(axis=1), max_support=ecsw_max_support, 
                                            rel_err_thresh=ecsw_err_thresh)
    weights = np.append(np.ones((5,)), weights1)


    # evaluate ROMs at mu_rom
    hdm_snaps = load_or_compute_snaps(mu_rom, grid, w0, dt, num_steps, snap_folder=snap_folder)
    hprom_snaps = inviscid_burgers_ecsw(grid, weights, w0, dt, num_steps, mu_rom, basis_trunc)

    fig, ax = plt.subplots()
    snaps_to_plot = range(num_steps//4, num_steps+1, num_steps//4)
    plot_snaps(grid, hdm_snaps, snaps_to_plot, 
               label='HDM', fig_ax=(fig,ax))
    plot_snaps(grid, hprom_snaps, snaps_to_plot, 
               label='HPROM', fig_ax=(fig,ax), color='red', linewidth=1, linestyle='dashed')

    ax.set_xlim([grid.min(), grid.max()])
    ax.set_xlabel('x')
    ax.set_ylabel('w')
    ax.set_title('Comparing HDM and ROMs')
    ax.legend()
    plt.show()

    pdb.set_trace()


if __name__ == "__main__":
    main()
