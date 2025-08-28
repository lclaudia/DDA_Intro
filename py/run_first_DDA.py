#!/usr/bin/env python3
"""
Exact Python translation of run_first_DDA.jl
DDA (Delay Differential Analysis) demonstration script
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import platform
from itertools import combinations
import seaborn as sns

# Import local functions
from dda_functions import (
    create_mod_nr as make_MOD_nr,
    create_model as make_MODEL,
    deriv_all,
)


# System parameters
NrSyst = 4

ROS = np.array(
    [[0, 0, 2], [0, 0, 3], [1, 0, 1], [1, 0, 2], [2, 0, 0], [2, 0, 3], [2, 1, 3]]
)

MOD_nr, DIM, ODEorder, P = make_MOD_nr(ROS, NrSyst)

# Model parameters
a123 = 0.21
a456 = 0.20

b1 = 0.2150
b2 = 0.2020
b4 = 0.4050
b5 = 0.3991

c = 5.7

MOD_par = np.array(
    [
        [-1, -1, 1, a123, b1, -c, 1],
        [-1, -1, 1, a123, b2, -c, 1],
        [-1, -1, 1, a456, b4, -c, 1],
        [-1, -1, 1, a456, b5, -c, 1],
    ]
)
MOD_par = MOD_par.T.flatten()

# Integration parameters
TRANS = 20000
dt = 0.05
# Set random seed for reproducible results matching Julia
np.random.seed(42)
X0 = np.random.rand(DIM * NrSyst)
CH_list = list(range(1, DIM * NrSyst + 1, DIM))
DELTA = 2
FN_DATA = "ROS_4.ascii"
L = 20000

# Use existing Julia-generated data for exact comparison
# If you want to generate new data, delete ROS_4.ascii file first
if not os.path.isfile(FN_DATA):
    print(
        "Error: ROS_4.ascii file not found. Please run Julia version first to generate data."
    )
    exit(1)

# Load data
Y = np.loadtxt(FN_DATA)

# DDA parameters
WL = 2000
WS = 1000
TAU = [32, 9]
dm = 4
order = 3
nr_delays = 2
TM = max(TAU)
WN = int(1 + np.floor((Y.shape[0] - (WL + TM + 2 * dm - 1)) / WS))

print(f"Data shape: {Y.shape}")
print(f"Number of windows: {WN}")

########  SingleTimeseries DDA  ########

Y = np.loadtxt(FN_DATA)
Y_single = Y[:, 0]  # First column only

ST = np.full((WN, 4), np.nan)
for wn in range(WN):
    anf = wn * WS
    ende = anf + WL + TM + 2 * dm - 1

    data = Y_single[anf : ende + 1]
    ddata = deriv_all(data, dm)
    data = data[dm:-dm]

    STD = np.std(data, ddof=1)  # Use sample std to match Julia
    DATA = (data - np.mean(data)) / STD
    dDATA = ddata / STD

    # CRITICAL FIX: Julia constructs M first, then slices dDATA
    # Julia: M = hcat(DATA[(TM+1:end).-TAU[1]], DATA[(TM+1:end).-TAU[2]], DATA[(TM+1:end).-TAU[1]] .^ 3)
    # Julia indexing: (TM+1:end).-TAU[1] = TM+1-TAU[1]:end-TAU[1]
    # Python 0-based: TM-TAU[0]:len(DATA)-TAU[0]
    M = np.column_stack(
        [
            DATA[TM - TAU[0] : len(DATA) - TAU[0]],  # First delay coordinate
            DATA[TM - TAU[1] : len(DATA) - TAU[1]],  # Second delay coordinate
            (DATA[TM - TAU[0] : len(DATA) - TAU[0]]) ** 3,  # Nonlinear term
        ]
    )

    # Julia: dDATA = dDATA[TM+1:end] (AFTER M construction)
    dDATA_sliced = dDATA[TM:]

    # Use solve instead of lstsq to match Julia's \ operator more closely
    try:
        ST[wn, :3] = np.linalg.solve(
            M.T @ M, M.T @ dDATA_sliced
        )  # Normal equation approach
    except np.linalg.LinAlgError:
        ST[wn, :3] = np.linalg.lstsq(M, dDATA_sliced, rcond=None)[0]
    ST[wn, 3] = np.sqrt(np.mean((dDATA_sliced - M @ ST[wn, :3]) ** 2))


###  for all time series

Y = np.loadtxt(FN_DATA)  # Reload full data
ST = np.full((WN, 4, Y.shape[1]), np.nan)

for n_Y in range(Y.shape[1]):
    for wn in range(WN):
        anf = wn * WS
        ende = anf + WL + TM + 2 * dm - 1

        data = Y[anf : ende + 1, n_Y]
        ddata = deriv_all(data, dm)
        data = data[dm:-dm]

        STD = np.std(data, ddof=1)  # Use sample std to match Julia
        DATA = (data - np.mean(data)) / STD
        dDATA = ddata / STD

        # Fixed matrix construction (same as single time series)
        M = np.column_stack(
            [
                DATA[TM - TAU[0] : len(DATA) - TAU[0]],
                DATA[TM - TAU[1] : len(DATA) - TAU[1]],
                (DATA[TM - TAU[0] : len(DATA) - TAU[0]]) ** 3,
            ]
        )

        # Julia: dDATA = dDATA[TM+1:end] (AFTER M construction)
        dDATA_sliced = dDATA[TM:]

        # Use solve instead of lstsq to match Julia's \ operator more closely
        try:
            ST[wn, :3, n_Y] = np.linalg.solve(M.T @ M, M.T @ dDATA_sliced)
        except np.linalg.LinAlgError:
            ST[wn, :3, n_Y] = np.linalg.lstsq(M, dDATA_sliced, rcond=None)[0]
        ST[wn, 3, n_Y] = np.sqrt(np.mean((dDATA_sliced - M @ ST[wn, :3, n_Y]) ** 2))


########  CrossTimeseries DDA  ########

NrCH = Y.shape[1]
CH = list(range(1, NrCH + 1))
LIST = list(combinations(CH, 2))
LIST = np.array(LIST) - 1  # Convert to 0-based indexing

CT = np.full((WN, 4, len(LIST)), np.nan)
for n_LIST in range(len(LIST)):
    ch1, ch2 = LIST[n_LIST]
    for wn in range(WN):
        anf = wn * WS
        ende = anf + WL + TM + 2 * dm - 1

        data1 = Y[anf : ende + 1, ch1]
        ddata1 = deriv_all(data1, dm)
        data1 = data1[dm:-dm]

        data2 = Y[anf : ende + 1, ch2]
        ddata2 = deriv_all(data2, dm)
        data2 = data2[dm:-dm]

        STD = np.std(data1, ddof=1)  # Use sample std to match Julia
        DATA1 = (data1 - np.mean(data1)) / STD
        dDATA1 = ddata1 / STD

        STD = np.std(data2, ddof=1)  # Use sample std to match Julia
        DATA2 = (data2 - np.mean(data2)) / STD
        dDATA2 = ddata2 / STD

        # Build matrices with fixed indexing
        M1 = np.column_stack(
            [
                DATA1[TM - TAU[0] : len(DATA1) - TAU[0]],
                DATA1[TM - TAU[1] : len(DATA1) - TAU[1]],
                (DATA1[TM - TAU[0] : len(DATA1) - TAU[0]]) ** 3,
            ]
        )

        M2 = np.column_stack(
            [
                DATA2[TM - TAU[0] : len(DATA2) - TAU[0]],
                DATA2[TM - TAU[1] : len(DATA2) - TAU[1]],
                (DATA2[TM - TAU[0] : len(DATA2) - TAU[0]]) ** 3,
            ]
        )

        # Slice derivatives AFTER matrix construction
        dDATA1_sliced = dDATA1[TM:]
        dDATA2_sliced = dDATA2[TM:]

        # Combine matrices and data
        M = np.vstack([M1, M2])
        dDATA_combined = np.concatenate([dDATA1_sliced, dDATA2_sliced])

        # Use solve instead of lstsq to match Julia's \ operator more closely
        try:
            CT[wn, :3, n_LIST] = np.linalg.solve(M.T @ M, M.T @ dDATA_combined)
        except np.linalg.LinAlgError:
            CT[wn, :3, n_LIST] = np.linalg.lstsq(M, dDATA_combined, rcond=None)[0]
        CT[wn, 3, n_LIST] = np.sqrt(
            np.mean((dDATA_combined - M @ CT[wn, :3, n_LIST]) ** 2)
        )


########  DynamicalErgodicity DDA  ########

st = np.mean(ST[:, -1, :], axis=0)  # Mean over windows, last column (errors)
ct = np.mean(CT[:, -1, :], axis=0)  # Mean over windows, last column (errors)

E = np.full((Y.shape[1], Y.shape[1]), np.nan)
for n_LIST in range(len(LIST)):
    ch1, ch2 = LIST[n_LIST]
    E[ch1, ch2] = abs(np.mean([st[ch1], st[ch2]]) / ct[n_LIST] - 1)
    E[ch2, ch1] = E[ch1, ch2]

# Create heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(E, annot=True, fmt=".2e", cmap="viridis")
plt.title("Dynamical Ergodicity Heatmap")
plt.savefig("ergodicity_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()


#################

nr_delays = 2
DDAmodel = np.array([[0, 0, 1], [0, 0, 2], [1, 1, 1]])
MODEL, L_AF, DDAorder = make_MODEL(DDAmodel)

FN_DDA = "ROS_4.DDA"

# Platform-specific executable handling
if platform.system() == "Windows":
    if not os.path.isfile("run_DDA_AsciiEdf.exe"):
        import shutil

        shutil.copy("run_DDA_AsciiEdf", "run_DDA_AsciiEdf.exe")
    CMD = ".\\run_DDA_AsciiEdf.exe"
else:
    CMD = "./run_DDA_AsciiEdf"

# Build command
CMD += " -ASCII"
CMD += f" -MODEL {' '.join(map(str, MODEL))}"
CMD += f" -TAU {' '.join(map(str, TAU))}"
CMD += f" -dm {dm} -order {DDAorder} -nr_tau {nr_delays}"
CMD += f" -DATA_FN {FN_DATA} -OUT_FN {FN_DDA}"
CMD += f" -WL {WL} -WS {WS}"
CMD += " -SELECT 1 1 0 0"
CMD += f" -CH_list {' '.join(map(str, LIST.flatten() + 1))}"
CMD += " -WL_CT 2 -WS_CT 2"

# Load results from external DDA computation
ST2 = np.loadtxt("ROS_4.DDA_ST")
ST2 = ST2[:, 2:]  # Skip first 2 columns

CT2 = np.loadtxt("ROS_4.DDA_CT")
CT2 = CT2[:, 2:]  # Skip first 2 columns

# Compare results and compute errors
# Julia uses column-major reshaping, Python uses row-major by default
# Need to match Julia's reshape behavior exactly
ST_reshaped = ST.reshape(WN, ST.shape[1] * ST.shape[2], order="F")
CT_reshaped = CT.reshape(WN, CT.shape[1] * CT.shape[2], order="F")

error_ST = np.mean(ST_reshaped - ST2)
error_CT = np.mean(CT_reshaped - CT2)

# Print with same precision as Julia for comparison
print(error_ST)
print(error_CT)
