"""
Use the Burgers equation to try out some learning-based hyper-reduction approaches
"""


import glob
import math
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import sklearn.cluster as clust
import pynndescent

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

def inviscid_burgers_LSPG(grid, w0, dt, num_steps, mu, basis):
    """
    Use a first-order Godunov spatial discretization and a second-order trapezoid rule
    time integrator to solve an LSPG PROM for a parameterized inviscid 1D burgers problem
    with a source term. The parameters are as follows:
    mu[0]: inlet state value
    mu[1]: the exponential rate of the exponential source term

    so the equation solved is
    w_t + (0.5 * w^2)_x = 0.02 * exp(mu[1]*x)
    w(x=grid[0], t) = mu[0]
    w(x, t=0) = w0
    """

    num_its = 0
    jac_time = 0
    res_time = 0
    ls_time = 0
    npod = basis.shape[1]
    snaps =  np.zeros((w0.size, num_steps+1))
    red_coords = np.zeros((npod, num_steps+1))
    y0 = basis.T.dot(w0)
    w0 = basis.dot(y0)
    snaps[:,0] = w0
    red_coords[:,0] = y0
    wp = w0.copy()
    yp = y0.copy()
    print("Running ROM of size {} for mu1={}, mu2={}".format(npod, mu[0], mu[1]))
    for i in range(num_steps):

        def res(w): 
            return inviscid_burgers_res(w, grid, dt, wp, mu)

        def jac(w):
            return inviscid_burgers_jac(w, grid, dt)

        print(" ... Working on timestep {}".format(i))
        y, resnorms, times = gauss_newton_LSPG(res, jac, basis, yp)
        jac_timep, res_timep, ls_timep = times
        num_its += len(resnorms)
        jac_time += jac_timep
        res_time += res_timep
        ls_time += ls_timep
        
        w = basis.dot(y)

        red_coords[:,i+1] = y.copy()
        snaps[:,i+1] = w.copy()
        wp = w.copy()
        yp = y.copy()

    return snaps, (num_its, jac_time, res_time, ls_time)

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

    num_its = 0
    jac_time = 0
    res_time = 0
    ls_time = 0
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
        y, resnorms, times = gauss_newton_LSPG(res, jac, basis, yp)
        jac_timep, res_timep, ls_timep = times
        num_its += len(resnorms)
        jac_time += jac_timep
        res_time += res_timep
        ls_time += ls_timep

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

    return snaps, (num_its, jac_time, res_time, ls_time)

def inviscid_burgers_LSPG_knn(grid, w0, dt, num_steps, mu, snaps, basis_size, index=None):
    """
    Use a first-order Godunov spatial discretization and a second-order trapezoid rule
    time integrator to solve a knn-local LSPG PROM for a parameterized inviscid 1D burgers problem
    with a source term. The parameters are as follows:
    mu[0]: inlet state value
    mu[1]: the exponential rate of the exponential source term

    so the equation solved is
    w_t + (0.5 * w^2)_x = 0.02 * exp(mu[1]*x)
    w(x=grid[0], t) = mu[0]
    w(x, t=0) = w0
    """

    jac_time = 0
    res_time = 0
    ls_time = 0
    tbasis = 0
    tproj = 0
    num_its = 0
    rom_snaps =  np.zeros((w0.size, num_steps+1))
    basis, basis_inds, tbasisp = get_knn_basis(w0, snaps, basis_size, index=index)
    tbasis += tbasisp
    t0 = time.time()
    y0 = basis.T.dot(w0)
    w0 = basis.dot(y0)
    tproj += time.time() - t0
    rom_snaps[:,0] = w0
    red_coords = [y0]
    basis_ind_list = []
    wp = w0.copy()
    yp = y0.copy()
    print(("Running knn ROM with k={} for mu1={}, mu2={}").format(basis_size, mu[0], mu[1]))
    for i in range(num_steps):

        def res(w): 
            return inviscid_burgers_res(w, grid, dt, wp, mu)

        def jac(w):
            return inviscid_burgers_jac(w, grid, dt)

        print(" ... Working on timestep {} with snapshots {}".format(i, np.sort(basis_inds)))
        y, resnorms, times = gauss_newton_LSPG(res, jac, basis, yp)
        jac_timep, res_timep, ls_timep = times
        num_its += len(resnorms)
        jac_time += jac_timep
        res_time += res_timep
        ls_time += ls_timep

        w = basis.dot(y)

        red_coords += [y.copy()]
        basis_ind_list += [basis_inds]
        rom_snaps[:,i+1] = w.copy()

        basis, basis_inds, tbasisp = get_knn_basis(w, snaps, basis_size, index=index)
        tbasis += tbasisp
        t0 = time.time()
        yp = basis.T.dot(w)
        wp = basis.dot(yp)
        tproj += time.time() - t0

    return rom_snaps, (tbasis, tproj, num_its, jac_time, res_time, ls_time)

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
    jac_time = 0
    res_time = 0
    ls_time = 0
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
        t0 = time.time()
        J = jac(w)
        jac_time += time.time() - t0
        t0 = time.time()
        f = func(w)
        res_time += time.time() - t0
        t0 = time.time()
        JV = J.dot(basis)
        dy, lst_res, rank, sval = np.linalg.lstsq(JV, -f, rcond=None)
        ls_time += time.time() - t0
        y += dy
        w = basis.dot(y)
        
    return y, resnorms, (jac_time, res_time, ls_time)

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

def compute_local_bases(snaps, num_clusts, energy_thresh=None, min_size=None,
                        max_size=None, overlap_frac=None):
    """ 
    Given a set of snapshots, cluster them and form POD bases for each set of
    clustered snapshots
    """
    kmeans = clust.KMeans(n_clusters=num_clusts, random_state=0)
    kmeans.fit(snaps.T)
    clust_inds = [np.where(kmeans.labels_ == i)[0] for i in range(num_clusts)]
    centroids = kmeans.cluster_centers_
    if overlap_frac is not None:
        clust_inds, centroids = apply_kmeans_overlap(clust_inds, centroids, snaps, overlap_frac=overlap_frac)
    local_bases = []
    for iclust in range(num_clusts):
        clust_snaps = snaps[:, clust_inds[iclust]]
        basis, sigma = POD(clust_snaps)
        num_vecs = podsize(sigma, energy_thresh=energy_thresh, min_size=min_size, max_size=max_size)
        local_bases += [ basis[:, :num_vecs] ]

    return local_bases, centroids

def apply_kmeans_overlap(clust_inds, centroids, snaps, overlap_frac=0.1):
    """ 
    Given a set of kmeans-assigned clust indices and snapshots, produce new cluster
    indices with overlap
    """  
    nclust = centroids.shape[0]
    nsnaps = snaps.shape[1]
    neighbs = [set() for i in range(nclust)]
    # build inter-cluster connectivity
    for isnap in range(nsnaps):
        snap = snaps[:, isnap]
        dists = np.array([np.linalg.norm(snap - centroids[i,:]) for i in range(nclust)])
        nearest_inds = np.argpartition(dists, 2)[:2]
        neighbs[nearest_inds[0]].add(nearest_inds[1])
        neighbs[nearest_inds[1]].add(nearest_inds[0])

    # augment clusters to add overlap
    new_clust_inds = []
    for iclust in range(nclust):
        new_clust_inds_i = set(clust_inds[iclust])
        for ineighb in neighbs[iclust]:
            num_neighb_snaps = clust_inds[ineighb].size
            num_overlap = int(math.ceil(num_neighb_snaps * overlap_frac))
            dists = np.array([np.linalg.norm(snaps[:,i] - centroids[iclust,:]) 
                              for i in clust_inds[ineighb]])
            nearest_inds = clust_inds[ineighb][np.argpartition(dists, num_overlap)[:num_overlap]]
            new_clust_inds_i = new_clust_inds_i | set(nearest_inds)
        new_clust_inds += [list(new_clust_inds_i)]

    # compute new centroids
    new_centroids = np.zeros_like(centroids)
    for iclust in range(nclust):
        clust_snaps = snaps[:, new_clust_inds[iclust]]
        new_centroids[iclust, :] = clust_snaps.mean(axis=1)

    return new_clust_inds, new_centroids

def nearest_centroid(w, centroids):
    """ Returns the index of the nearest centroid to the state w """
    num_clusts = centroids.shape[0]
    dists = [np.linalg.norm(w - centroids[iclust, :]) for iclust in range(num_clusts)]
    inearest = np.array(dists).argmin()
    return inearest

def get_knn_basis(w, snaps, basis_size, index=None):
    """ Returns an orthonormal basis spanning the space of the nearest snapshots to w """
    t0 = time.time()
    if index is None:
        diff = np.expand_dims(w, axis=1) - snaps
        dists = np.linalg.norm(diff, axis=0)
        nearest_inds = np.argpartition(dists, basis_size)[:basis_size]
    else:
        nearest_inds, dists = index.query(np.expand_dims(w, axis=0), k=basis_size,
                                          epsilon=0.5)
        nearest_inds = nearest_inds.squeeze()
    nearest_snaps = snaps[:, nearest_inds]
    q, r  = np.linalg.qr(nearest_snaps)
    tbasis = time.time() - t0
    return q, nearest_inds, tbasis

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

def compute_error(rom_snaps, hdm_snaps):
    """ Computes the relative error at each timestep """
    sq_hdm = np.sqrt(np.square(rom_snaps).sum(axis=0))
    sq_err = np.sqrt(np.square(rom_snaps - hdm_snaps).sum(axis=0))
    rel_err = sq_err / sq_hdm
    return rel_err, rel_err.mean()

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
        snaps = np.load(snap_fn)[:, :num_steps+1]
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
               label='PROM', fig_ax=(fig,ax1), color='blue', linewidth=1)
    plot_snaps(grid, local_rom_snaps, snaps_to_plot, 
               label='Local PROM, {} clusts, {} avg. basis size'.format(num_clusts, avg_npod), 
               fig_ax=(fig,ax1), color='red', linewidth=1)
    plot_snaps(grid, knn_rom_snaps, snaps_to_plot, 
               label='knn PROM, basis size {}'.format(num_knn), 
               fig_ax=(fig,ax1), color='orange', linewidth=1)

    ax1.set_xlim([grid.min(), grid.max()])
    ax1.set_xlabel('x')
    ax1.set_ylabel('w')
    ax1.set_title('Comparing HDM and ROMs')
    ax1.legend(loc='upper left')

    ax2.plot(errors, label='PROM', color='blue')
    ax2.plot(local_errors, 
             label='Local PROM, {} clusts, {} avg. basis size'.format(num_clusts, avg_npod), 
             color='red')
    ax2.plot(knn_errors, 
             label='knn PROM, basis size {}'.format(num_clusts), 
             color='orange')
    ax2.set_xlabel('Time index')
    ax2.set_ylabel('Relative error')
    ax2.set_title('Comparing relative error')
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
