"""
Run the burgers' equation with knn PROMs
"""

import glob
import math
import time

import numpy as np
import matplotlib.pyplot as plt
import pynndescent

from hypernet import (
                      make_1D_grid,
                      load_or_compute_snaps,
                      POD,
                      podsize,
                      compute_local_bases,
                      inviscid_burgers_LSPG_local,
                      inviscid_burgers_LSPG_knn,
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
    num_knn = 10

    dt = 0.07
    num_steps_offline = 500
    num_steps_eval = 400
    num_cells = 500
    xl, xu = 0, 100
    w0 = np.ones(num_cells)
    grid = make_1D_grid(xl, xu, num_cells)

    mu_samples = [
               [4.3, 0.021],
               [5.1, 0.030]
              ]
    # mu_rom = [4.3, 0.021]
    mu_rom = [4.7, 0.026]

    # Generate or retrieve HDM snapshots
    all_snaps_list = []
    for mu in mu_samples:
        snaps = load_or_compute_snaps(mu, grid, w0, dt, num_steps_offline, snap_folder=snap_folder)
        all_snaps_list += [snaps]

    snaps = np.hstack(all_snaps_list)   

    # construct basis using mu_samples params
    basis, sigma = POD(snaps)
    num_vecs = podsize(sigma, energy_thresh=energy_thresh)
    basis_trunc = basis[:, :num_vecs]
    local_bases, centroids = compute_local_bases(snaps, num_clusts, 
                                                 energy_thresh=energy_thresh_local, 
                                                 min_size=min_size, max_size=max_size,
                                                 overlap_frac=overlap_frac)
    sum_npod = sum(basis_i.shape[1] for basis_i in local_bases)
    avg_npod = sum_npod / len(local_bases)
    # prepare the nearest-neighbor search index
    print("Setting up nearest-neighbor index")
    index = pynndescent.NNDescent(snaps.T, n_neighbors=200)
    index.prepare()

    # evaluate ROM at mu_rom
    t0 = time.time()
    knn_rom_snaps, times = inviscid_burgers_LSPG_knn(grid, w0, dt, num_steps_eval, mu_rom,
                                                  snaps, num_knn, index=index)
    t_knn = time.time() - t0
    tbasis, tproj, its_knn, jac_time_knn, res_time_knn, ls_time_knn = times

    t0 = time.time()
    local_rom_snaps, times = inviscid_burgers_LSPG_local(grid, w0, dt, num_steps_eval, mu_rom,
                                                  local_bases, centroids)
    t_loc = time.time() - t0
    its_loc, jac_time_loc, res_time_loc, ls_time_loc = times

    t0 = time.time()
    rom_snaps, times = inviscid_burgers_LSPG(grid, w0, dt, num_steps_eval, mu_rom, basis_trunc)
    t_rom = time.time() - t0
    its_rom, jac_time_rom, res_time_rom, ls_time_rom = times

    hdm_snaps = load_or_compute_snaps(mu_rom, grid, w0, dt, num_steps_eval, snap_folder=snap_folder)
    errors, rms_err = compute_error(rom_snaps, hdm_snaps)
    local_errors, local_rms_err = compute_error(local_rom_snaps, hdm_snaps)
    knn_errors, knn_rms_err = compute_error(knn_rom_snaps, hdm_snaps)

    fig, (ax1, ax2) = plt.subplots(2)
    snaps_to_plot = range(num_steps_eval//10, num_steps_eval+1, num_steps_eval//10)
    plot_snaps(grid, hdm_snaps, snaps_to_plot, 
               label='HDM', fig_ax=(fig,ax1))
    plot_snaps(grid, rom_snaps, snaps_to_plot, 
               label='PROM, basis size {}'.format(num_vecs), 
               fig_ax=(fig,ax1), color='blue', linewidth=1)
    plot_snaps(grid, local_rom_snaps, snaps_to_plot, 
               label='Local PROM, {} clusts, {} avg. basis size'.format(num_clusts, avg_npod), 
               fig_ax=(fig,ax1), color='red', linewidth=1)
    plot_snaps(grid, knn_rom_snaps, snaps_to_plot, 
               label='knn PROM, basis size {}'.format(num_knn), 
               fig_ax=(fig,ax1), color='orange', linewidth=1)

    ax1.set_xlim([grid.min(), grid.max()])
    ax1.set_ylim([1, 7])
    ax1.set_xlabel('x', fontsize=15)
    ax1.set_ylabel('w', fontsize=15)
    ax1.set_title('Comparing HDM and ROMs', fontsize=15)
    ax1.legend(loc='upper left')

    ax2.plot(errors, 
            label='PROM, basis size {}'.format(num_vecs), 
            color='blue')
    ax2.plot(local_errors, 
             label='Local PROM, {} clusts, {} avg. basis size'.format(num_clusts, avg_npod), 
             color='red')
    ax2.plot(knn_errors, 
             label='knn PROM, basis size {}'.format(num_clusts), 
             color='orange')
    ax2.set_xlabel('Time index', fontsize=15)
    ax2.set_ylabel('Relative error', fontsize=15)
    ax2.set_title('Comparing relative error', fontsize=15)
    print('PROM rel. error:        {}'.format(rms_err))
    print('Local PROM rel. error:  {}'.format(local_rms_err))
    print('knn PROM rel. error:    {}'.format(knn_rms_err))
    print('--------------------------')
    print(('PROM time:       {:.4f}, ' + 
                            '{} its, ' +
                            '{:.4f} jac, ' +
                            '{:.4f} res, ' +
                            '{:.4f} LS').format(t_rom, its_rom, jac_time_rom,
                                                res_time_rom, ls_time_rom))
    print(('Local PROM time: {:.4f}, ' +
                            '{} its, ' +
                            '{:.4f} jac, ' +
                            '{:.4f} res, ' +
                            '{:.4f} LS').format(t_loc, its_loc, jac_time_loc,
                                                res_time_loc, ls_time_loc))
    print(('knn PROM time:   {:.4f}, ' +
                           '{} its, ' +
                           '{:.4f} jac, ' +
                           '{:.4f} res, ' +
                           '{:.4f} LS, ' +
                           '{:.4f} making basis, ' + 
                           '{:.4f} projections').format(t_knn, its_knn, jac_time_knn,
                                                        res_time_knn, ls_time_knn, tbasis, tproj))
    ax2.legend(loc='upper right')
    fig.tight_layout()
    plt.show()

    pdb.set_trace()


if __name__ == "__main__":
    main()
