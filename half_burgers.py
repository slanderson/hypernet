"""
Produce a comparison of row ROBs and ordinary POD using the burgers' equation
"""

import numpy as np
import matplotlib.pyplot as plt
from hypernet import (
        make_1D_grid, 
        inviscid_burgers_implicit, 
        POD,
        podsize,
        inviscid_burgers_LSPG,
        plot_snaps
        )

import pdb

def main():
    energy = 0.999999
    dt = 0.05
    num_steps = 400
    num_cells = 500
    xl, xu = 0, 100
    mu = np.array([4.0, 0.000])

    grid = make_1D_grid(xl, xu, num_cells)
    w0 = np.ones(num_cells)
    w0[:num_cells//2] = 4.0 + 0.005 * grid[:num_cells//2]

    snaps = inviscid_burgers_implicit(grid, w0, dt, num_steps, mu)

    # construct basis using mu_samples params
    basis, sigma = POD(snaps)
    num_vecs_pod = podsize(sigma, energy_thresh=energy)
    basis_trunc = basis[:, :num_vecs_pod]

    basis_left, sigma_left = POD(snaps[:250, :])
    basis_right, sigma_right = POD(snaps[250:, :])
    sigma_combined = np.hstack((sigma_left, sigma_right)).tolist()
    clust_assign = [0]*250 + [1]*250
    sigma_sorted, clusts_sorted = zip(*sorted(zip(sigma_combined, clust_assign), reverse=True))
    podsize_rows = podsize(np.array(sigma_sorted), energy_thresh=energy)
    podsize_right = sum(clusts_sorted[:podsize_rows])
    podsize_left = podsize_rows - podsize_right
    basis_rows = np.zeros((num_cells, podsize_rows))
    basis_rows[:250, :podsize_left] = basis_left[:, :podsize_left]
    basis_rows[250:, podsize_left:] = basis_right[:, :podsize_right]

    rom_snaps, rom_times = inviscid_burgers_LSPG(grid, w0, dt, num_steps, mu, basis_trunc)
    row_snaps, row_times = inviscid_burgers_LSPG(grid, w0, dt, num_steps, mu, basis_rows)

    snaps_to_plot = range(0, 279, 70)
    snaps_to_plot = [0, 100, 200, 300]
    fig, ax = plt.subplots()
    fig, ax = plot_snaps(grid, snaps, snaps_to_plot, fig_ax=(fig,ax), 
                         color='black', linewidth=3, linestyle='dashed',
                         label='High-dimensional model')
    fig, ax = plot_snaps(grid, rom_snaps, snaps_to_plot, fig_ax=(fig,ax), 
                         color='red', linewidth=1, 
                         label='LSPG w/ conventional POD ROB')
    fig, ax = plot_snaps(grid, row_snaps, snaps_to_plot, fig_ax=(fig,ax), 
                         color='blue', linewidth=1, 
                         label='LSPG w/ spatially local ROB')
    ax.set_xlim([grid.min(), grid.max()])
    ax.set_xlabel('x', fontsize=13)
    ax.set_ylabel('w', fontsize=13)
    ax.set_title('Comparing spatially local and conventional POD bases', fontsize=15)
    ax.text(51, 1.2, '$t=0$')
    ax.text(65, 1.3, '$t=5$')
    ax.text(79, 1.4, '$t=10$')
    ax.text(93, 1.5, '$t=15$')
    ax.legend()
    plt.show()

    pdb.set_trace()


if __name__ == "__main__":
    main()
