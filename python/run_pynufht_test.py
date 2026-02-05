#!/usr/bin/env python3
import os
import numpy as np
from math import gamma

# import built extension
import _pynufht as pn

thisdir = os.path.dirname(__file__)
gl_path = os.path.join(thisdir, '..', 'data', 'gl1000')
if not os.path.exists(gl_path):
    raise RuntimeError('GL file not found: ' + gl_path)

gl = np.loadtxt(gl_path)
absc = gl[:,0].copy()
wght = gl[:,1].copy()

# rescale from [-1,1] to [0, rmax]
rmax = 10.0
absc = (rmax/2.0) * (absc + 1.0)
wght *= (rmax/2.0)

mu = 0.0
m = len(absc)
cs = (absc**(mu+1.0)) * np.exp(-0.5 * absc**2) * wght

Nk = 100
kmax = 10.0
k_grid = kmax * (np.arange(1, Nk+1) / Nk)

# call binding
g = pn.nufht(mu, absc, cs, k_grid)

analytic = gamma(mu+1.0) * (k_grid**mu) * np.exp(-0.5 * k_grid**2)
rel_err = np.abs(g - analytic) / (np.abs(analytic) + 1e-16)
for k,a,nr,re in zip(k_grid, analytic, g, rel_err):
    print(f"{k:12.6e}  {a:15.8e}  {nr:15.8e}  {re:12.3e}")

print(f"max abs err = {np.max(np.abs(g-analytic)):.3e}, max rel err = {np.max(rel_err):.3e}")
