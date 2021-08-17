"""
Use the Burgers equation to try out some learning-based hyper-reduction approaches
"""


import glob

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import sklearn.cluster as clust

from lsqnonneg import lsqnonneg

import pdb


def make_1D_grid(x_low, x_up, num_cells):
    """
    Returns a 1d ndarray of cell boundary points between a lower bound and an upper bound
    with the given number of cells
    """
    grid = np.linspace(x_low, x_up, num_cells+1)
    return grid

def inviscid_burgers_explicit(grid, w0, dt, num_steps, mu):
    """
    Use a first-order Godunov spatial discretization and a first-order forward Euler time
    integrator to solve a parameterized inviscid 1D burgers problem with a source term .
    The parameters
    are as follows:
    mu[0]: inlet state value
    mu[1]: the exponential rate of the exponential source term

    so the equation solved is
    w_t + (0.5 * w^2)_x = 0.02 * exp(mu[1]*x)
    w(x=grid[0], t) = mu[0]
    w(x, t=0) = w0
    """

    snaps =  np.zeros((w0.size, num_steps+1))
    snaps[:,0] = w0
    wp = w0.copy()
    dx = grid[1:] - grid[:-1]
    xc = (grid[1:] + grid[:-1])/2
    f = np.zeros(grid.size)
    for i in range(num_steps):
        f[0] = 0.5 * mu[0]**2 
        f[1:] = 0.5 * np.square(wp)
        w = wp + dt*0.02*np.exp(mu[1]*xc) - dt*(f[1:] - f[:-1])/dx
        snaps[:,i+1] = w
        wp = w.copy()
    return snaps

def inviscid_burgers_implicit(grid, w0, dt, num_steps, mu):
    """
    Use a first-order Godunov spatial discretization and a second-order trapezoid rule
    time integrator to solve a parameterized inviscid 1D burgers problem with a source
    term.  The parameters are as follows:
    mu[0]: inlet state value
    mu[1]: the exponential rate of the exponential source term

    so the equation solved is
    w_t + (0.5 * w^2)_x = 0.02 * exp(mu[1]*x)
    w(x=grid[0], t) = mu[0]
    w(x, t=0) = w0
    """

    print("Running HDM for mu1={}, mu2={}".format(mu[0], mu[1]))
    snaps =  np.zeros((w0.size, num_steps+1))
    snaps[:,0] = w0
    wp = w0.copy()
    for i in range(num_steps):

        def res(w): 
            return inviscid_burgers_res(w, grid, dt, wp, mu)

        def jac(w):
            return inviscid_burgers_jac(w, grid, dt)

        print(" ... Working on timestep {}".format(i))
        w, resnorms = newton_raphson(res, jac, wp, max_its=50)

        snaps[:,i+1] = w.copy()
        wp = w.copy()

    return snaps

def inviscid_burgers_LSPG_local(grid, w0, dt, num_steps, mu, local_bases, centroids):
    """
    Use a first-order Godunov spatial discretization and a second-order trapezoid rule
    time integrator to solve a local LSPG PROM for a parameterized inviscid 1D burgers problem
    with a source term. The parameters are as follows:
    mu[0]: inlet state value
    mu[1]: the exponential rate of the exponential source term

    so the equation solved is
    w_t + (0.5 * w^2)_x = 0.02 * exp(mu[1]*x)
    w(x=grid[0], t) = mu[0]
    w(x, t=0) = w0
    """

    nclusts = len(local_bases) 
    npod_sum = sum(basis_i.shape[1] for basis_i in local_bases)
    npod_avg = npod_sum / nclusts
    snaps =  np.zeros((w0.size, num_steps+1))
    iclust = nearest_centroid(w0, centroids)
    basis = local_bases[iclust]
    y0 = basis.T.dot(w0)
    w0 = basis.dot(y0)
    snaps[:,0] = w0
    red_coords = [y0]
    iclusts = [iclust]
    wp = w0.copy()
    yp = y0.copy()
    print(("Running local ROM with {} clusters and avg. basis size {} "+
           "for mu1={}, mu2={}").format(nclusts, npod_avg, mu[0], mu[1]))
    for i in range(num_steps):

        def res(w): 
            return inviscid_burgers_res(w, grid, dt, wp, mu)

        def jac(w):
            return inviscid_burgers_jac(w, grid, dt)

        print(" ... Working on timestep {} using cluster {}".format(i, iclust))
        y, resnorms = gauss_newton_LSPG(res, jac, basis, yp)
        w = basis.dot(y)

        red_coords += [y.copy()]
        snaps[:,i+1] = w.copy()
        wp = w.copy()
        yp = y.copy()

        iclust = nearest_centroid(w, centroids)
        iclusts += [iclust]
        if iclusts[-2] != iclusts[-1]:
            basis = local_bases[iclust]
            yp = basis.T.dot(w)
            wp = basis.dot(yp)

    return snaps

def inviscid_burgers_ecsw(grid, weights, w0, dt, num_steps, mu, basis):
    """
    Use a first-order Godunov spatial discretization and a second-order trapezoid rule
    time integrator to solve an ECSW HPROM for a parameterized inviscid 1D burgers problem
    with a source term. The parameters are as follows:
    mu[0]: inlet state value
    mu[1]: the exponential rate of the exponential source term

    so the equation solved is
    w_t + (0.5 * w^2)_x = 0.02 * exp(mu[1]*x)
    w(x=grid[0], t) = mu[0]
    w(x, t=0) = w0
    """

    npod = basis.shape[1]
    snaps =  np.zeros((w0.size, num_steps+1))
    red_coords = np.zeros((npod, num_steps+1))
    y0 = basis.T.dot(w0)
    w0 = basis.dot(y0)
    snaps[:,0] = w0
    red_coords[:,0] = y0
    wp = w0.copy()
    wtmp = np.zeros_like(w0)
    yp = y0.copy()
    sample_inds, = np.where(weights != 0)
    sample_weights = weights[sample_inds]
    nsamp = sample_weights.size

    print("Running HROM of size {} with {} sample nodes for mu1={}, mu2={}".format(npod, nsamp, mu[0], mu[1]))
    for i in range(num_steps):

        def res(w): 
            # return inviscid_burgers_ecsw_res(w, grid, sample_inds, dt, wp, mu)
            return inviscid_burgers_res(w, grid, dt, wp, mu)

        def jac(w):
            # return inviscid_burgers_ecsw_jac(w, grid, sample_inds, dt)
            return inviscid_burgers_jac(w, grid, dt)

        print(" ... Working on timestep {}".format(i))
        y, resnorms = gauss_newton_ECSW(res, jac, basis, yp, wtmp, sample_inds, sample_weights)
        w = basis.dot(y)

        red_coords[:,i+1] = y.copy()
        snaps[:,i+1] = w.copy()
        wp = w.copy()
        yp = y.copy()

    return snaps

def inviscid_burgers_ecsw_res(w, grid, sample_inds, dt, wp, mu):
    """ 
    Returns a residual vector for the ECSW hyper-reduced 1d inviscid burgers equation
    using a first-order Godunov space discretization and a 2nd-order trapezoid rule time
    integrator
    Note: sample_inds must be sorted! (if the left boundary is included)
    """
    dx = grid[sample_inds+1] - grid[sample_inds]
    xc = (grid[sample_inds+1] + grid[sample_inds])/2

    fl = np.zeros(sample_inds.size)
    fr = np.zeros(sample_inds.size)
    flp = np.zeros(sample_inds.size)
    frp = np.zeros(sample_inds.size)
    fr = 0.5 * np.square(w[sample_inds])
    frp = 0.5 * np.square(wp[sample_inds])
    if 0 in sample_inds:
        fl[0] = 0.5 * mu[0]**2
        flp[0] = 0.5 * mu[0]**2
        fl[1:] = 0.5 * np.square(w[sample_inds[1:]-1])
        flp[1:] = 0.5 * np.square(wp[sample_inds[1:]-1])
    else:
        fl = 0.5 * np.square(w[sample_inds-1])
        flp = 0.5 * np.square(wp[sample_inds-1])

    src = dt*0.02*np.exp(mu[1]*xc)
    r = w[sample_inds] - wp[sample_inds] - src + 0.5*(dt/dx)*( (frp-flp) + (fr-fl) )
    return r

def inviscid_burgers_ecsw_jac(w, grid, sample_inds, dt):
    """ 
    Returns a Jacobian for the ECSW hyper-reduced 1d inviscid burgers equation
    using a first-order Godunov space discretization and a 2nd-order trapezoid rule time
    integrator
    """
    n_samp = sample_inds.size
    dx = grid[sample_inds+1] - grid[sample_inds]
    xc = (grid[sample_inds+1] + grid[sample_inds])/2

    J = sp.lil_matrix((n_samp, n_samp))
    J += sp.eye(n_samp)
    J += 0.5*sp.diags( (dt/dx)*w[sample_inds])
    for i, ind in enumerate(sample_inds):
        if ind+1 in sample_inds:
            J[i+1, i] = -0.5*w[ind]*dt/dx[i+1]
    return J.tocsr()
    ## test credential manager    

def inviscid_burgers_res(w, grid, dt, wp, mu):
    """ 
    Returns a residual vector for the 1d inviscid burgers equation using a first-order
    Godunov space discretization and a 2nd-order trapezoid rule time integrator
    """
    dx = grid[1:] - grid[:-1]
    xc = (grid[1:] + grid[:-1])/2

    f = np.zeros(grid.size)
    fp = np.zeros(grid.size)
    f[0] = 0.5 * mu[0]**2 
    f[1:] = 0.5 * np.square(w)
    fp[0] = 0.5 * mu[0]**2 
    fp[1:] = 0.5 * np.square(wp)
    src = dt*0.02*np.exp(mu[1]*xc)
    r = w - wp - src + 0.5*(dt/dx)*((fp[1:]-fp[:-1]) + (f[1:] - f[:-1]))
    return r

def inviscid_burgers_jac(w, grid, dt):
    """ 
    Returns a sparse Jacobian for the 1d inviscid burgers equation using a first-order
    Godunov space discretization and a 2nd-order trapezoid rule time integrator
    """
    dx = grid[1:] - grid[:-1]
    xc = (grid[1:] + grid[:-1])/2

    J = sp.lil_matrix((xc.size, xc.size))
    J += sp.eye(xc.size)
    J += 0.5*sp.diags( (dt/dx)*w )
    J -= 0.5*sp.diags( (dt/dx[1:])*w[:-1] , -1)
    return J.tocsr()

def newton_raphson(func, jac, x0, max_its=20, relnorm_cutoff=1e-12):
    x = x0.copy()
    init_norm = np.linalg.norm(func(x0))
    resnorms = []
    for i in range(max_its):
        resnorm = np.linalg.norm(func(x))
        resnorms += [resnorm]
        if resnorm/init_norm < relnorm_cutoff:
            break
        J = jac(x)
        f = func(x)
        x -= sp.linalg.spsolve(J, f)
    return x, resnorms

def gauss_newton_LSPG(func, jac, basis, y0, 
                      max_its=20, relnorm_cutoff=1e-5, min_delta=0.1):
    y = y0.copy()
    w = basis.dot(y0)
    init_norm = np.linalg.norm(func(w))
    resnorms = []
    for i in range(max_its):
        resnorm = np.linalg.norm(func(w))
        resnorms += [resnorm]
        if resnorm/init_norm < relnorm_cutoff:
            break
        if (len(resnorms) > 1) and (abs((resnorms[-2] - resnorms[-1]) / resnorms[-2]) < min_delta):
            break
        J = jac(w)
        f = func(w)
        JV = J.dot(basis)
        dy, lst_res, rank, sval = np.linalg.lstsq(JV, -f, rcond=None)
        y += dy
        w = basis.dot(y)
        
    return y, resnorms

def gauss_newton_ECSW(func, jac, basis, y0, w, sample_inds, sample_weights,
                      stepsize=1, max_its=20, relnorm_cutoff=1e-4, min_delta=1E-8):
    y = y0.copy()
    w = basis.dot(y0)
    init_norm = np.linalg.norm(func(w)[sample_inds] * sample_weights)
    resnorms = []
    for i in range(max_its):
        resnorm = np.linalg.norm(func(w)[sample_inds] * sample_weights)
        resnorms += [resnorm]
        if resnorm/init_norm < relnorm_cutoff:
            break
        if (len(resnorms) > 1) and (abs((resnorms[-2] - resnorms[-1]) / resnorms[-2]) < min_delta):
            break

        J = jac(w).toarray()
        JV = J.dot(basis)[sample_inds, :]
        JVw = np.diag(sample_weights).dot(JV)

        f = func(w)[sample_inds]
        fw = f * sample_weights
        dy = np.linalg.lstsq(JVw, -fw, rcond=None)[0]
        # redjac = JV.T.dot(JV)
        # fred = JV.T.dot(f)
        # dy = np.linalg.solve(redjac, -fred)
        y += stepsize*dy
        w = basis.dot(y)

    return y, resnorms

def POD(snaps):
    u, s, vh = np.linalg.svd(snaps, full_matrices=False)
    return u, s

def podsize(svals, energy_thresh=None, min_size=None, max_size=None):
    """ Returns the number of vectors in a basis that meets the given criteria """

    if (energy_thresh is None) and (min_size is None) and (max_size is None):
        raise RuntimeError('Must specify at least one truncation criteria in podsize()')

    if energy_thresh is not None:
        svals_squared = np.square(svals.copy())
        energies = np.cumsum(svals_squared)
        energies /= np.square(svals).sum()
        numvecs = np.where(energies >= energy_thresh)[0][0]
    else:
        numvecs = min_size

    if min_size is not None and numvecs < min_size:
        numvecs = min_size

    if max_size is not None and numvecs > max_size:
        numvecs = max_size

    return numvecs

def compute_local_bases(snaps, num_clusts, energy_thresh):
    """ 
    Given a set of snapshots, cluster them and form POD bases for each set of
    clustered snapshots
    """
    kmeans = clust.KMeans(n_clusters=num_clusts, random_state=0)
    kmeans.fit(snaps.T)
    clust_assignments = kmeans.labels_
    centroids = kmeans.cluster_centers_
    local_bases = []
    for iclust in range(num_clusts):
        clust_snaps = snaps[:, clust_assignments == iclust]
        basis, sigma = POD(clust_snaps)
        num_vecs = podsize(sigma, energy_thresh=energy_thresh)
        local_bases += [ basis[:, :num_vecs] ]

    return local_bases, centroids

def nearest_centroid(w, centroids):
    """ Returns the index of the nearest centroid to the state w """
    num_clusts = centroids.shape[0]
    dists = [np.linalg.norm(w - centroids[iclust, :]) for iclust in range(num_clusts)]
    inearest = np.array(dists).argmin()
    return inearest

def compute_ECSW_training_matrix(snaps, prev_snaps, basis, res, jac, grid, dt, mu):
    """
    Assembles the ECSW hyper-reduction training matrix.  Running a non-negative least
    squares algorithm with an early stopping criteria on these matrices will give the
    sample nodes and weights
    This assumes the snapshots are for scalar-valued state variables
    """
    n_hdm, n_snaps = snaps.shape
    n_pod = basis.shape[1]
    C = np.zeros((n_pod * n_snaps, n_hdm))
    for isnap in range(1,n_snaps):
        snap = prev_snaps[:, isnap]
        uprev = prev_snaps[:, isnap]
        u_proj = (basis.dot(basis.T)).dot(snap)
        ires = res(snap, grid, dt, uprev, mu)
        Ji = jac(snap, grid, dt)
        Wi = Ji.dot(basis)
        rki = Wi.T.dot(ires)
        for inode in range(n_hdm):
            C[isnap*n_pod:isnap*n_pod+n_pod, inode] = ires[inode]*Wi[inode]

    return C

def param_to_snap_fn(mu, snap_folder="param_snaps", suffix='.npy'):
    npar = len(mu)
    snapfn = snap_folder + '/'
    for i in range(npar):
        if i > 0:
            snapfn += '+'
        param_str = 'mu{}_{}'.format(i+1, mu[i])
        snapfn += param_str
    return snapfn + suffix

def get_saved_params(snap_folder="param_snaps"):
    param_fn_set = set(glob.glob(snap_folder+'/*'))
    return param_fn_set

def load_or_compute_snaps(mu, grid, w0, dt, num_steps, snap_folder="param_snaps"):
    snap_fn = param_to_snap_fn(mu, snap_folder=snap_folder)
    saved_params = get_saved_params(snap_folder=snap_folder)
    if snap_fn in saved_params:
        print("Loading saved snaps for mu1={}, mu2={}".format(mu[0], mu[1]))
        snaps = np.load(snap_fn)
    else:
        snaps = inviscid_burgers_implicit(grid, w0, dt, num_steps, mu)
        np.save(snap_fn, snaps)
    return snaps

def plot_snaps(grid, snaps, snaps_to_plot, linewidth=2, color='black', linestyle='solid', 
               label=None, fig_ax=None):
    if (fig_ax is None):
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax

    x = (grid[1:] + grid[:-1])/2
    is_first_line = True
    for ind in snaps_to_plot:
        if is_first_line:
            label2 = label
            is_first_line = False
        else:
            label2 = None
        ax.plot(x, snaps[:,ind], 
                color=color, linestyle=linestyle, linewidth=linewidth, label=label2)

    return fig, ax


def main():

    snap_folder = 'param_snaps'
    num_clusts = 5
    energy_thresh = 0.999999

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
    num_vecs = podsize(sigma, energy_thresh=energy_thresh)
    basis_trunc = basis[:, :num_vecs]
    local_bases, centroids = compute_local_bases(snaps, num_clusts, energy_thresh)

    # evaluate ROM at mu_rom
    local_rom_snaps = inviscid_burgers_LSPG_local(grid, w0, dt, num_steps, mu_rom,
                                                  local_bases, centroids)
    rom_snaps = inviscid_burgers_LSPG(grid, w0, dt, num_steps, mu_rom, basis_trunc)
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
