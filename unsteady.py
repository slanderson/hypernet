"""
Run the HDM, comparing implicit and explicit methods to debug. 
"""

import numpy as np
import matplotlib.pyplot as plt
from hypernet import inviscid_burgers_explicit, inviscid_burgers_implicit, make_1D_grid, plot_snaps

import pdb

def main():
    # first do an explicit solve with a small timestep, as a debugging reference
    dt = 0.007
    num_steps = 5000
    num_cells = 500
    xl, xu = 0, 100
    w0 = np.ones(num_cells)
    mu = np.array([4.3, 0.021])

    grid = make_1D_grid(xl, xu, num_cells)
    snaps = inviscid_burgers_explicit(grid, w0, dt, num_steps, mu)

    fig, ax = plt.subplots()
    snaps_to_plot = range(500, 5001, 500)
    fig, ax = plot_snaps(grid, snaps, snaps_to_plot, fig_ax=(fig, ax), 
                         linewidth=2, label='explicit')

    # compare implicit solve
    dt = 0.07
    num_steps = 500
    num_cells = 500
    w0 = np.ones(num_cells)

    grid = make_1D_grid(xl, xu, num_cells)
    snaps = inviscid_burgers_implicit(grid, w0, dt, num_steps, mu)

    snaps_to_plot = range(50, 501, 50)
    fig, ax = plot_snaps(grid, snaps, snaps_to_plot, fig_ax=(fig,ax), 
                         linestyle='dashed', color='red', label='implicit')
    ax.set_xlim([grid.min(), grid.max()])
    ax.set_xlabel('x')
    ax.set_ylabel('w')
    ax.set_title('Burgers Equation Snapshots')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
