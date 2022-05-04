"""
Define problem constants / 'magic numbers'
"""

import numpy as np
from hypernet import make_1D_grid

BATCH_SIZE = 16
HNET_BATCH_SIZE = 128
TRAIN_FRAC = 0.8
SNAP_FOLDER = "param_snaps"

ROM_SIZE = 40

SEED = 1234557

## PROBLEM-WIDE CONSTANTS
#   These define the underlying hdm, so set them once and use the same values for all ROM
#   and neural network runs
DT = 0.07
NUM_STEPS = 500
NUM_CELLS = 512
XL, XU = 0, 100
W0 = np.ones(NUM_CELLS)
GRID = make_1D_grid(XL, XU, NUM_CELLS)
MU1_RANGE = 4.25, 5.5
MU2_RANGE = 0.015, 0.03
SAMPLES_PER_MU = 3




