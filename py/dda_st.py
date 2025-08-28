"""
Single Timeseries DDA Analysis Module

This module provides functions for performing DDA (Delay Differential Analysis)
on single timeseries data.
"""

import numpy as np
from typing import Tuple, Optional
from numpy.typing import NDArray

from dda_functions import deriv_all


def compute_st_single(
    data: NDArray,
    TAU: list,
    dm: int = 4,
    order: int = 3,
    WL: int = 2000,
    WS: int = 1000,
) -> NDArray:
    """
    Compute single timeseries DDA structure coefficients.

    Args:
        data: Input timeseries data (1D array)
        TAU: List of two delay values [tau1, tau2]
        dm: Derivative method parameter (default: 4)
        order: Order of DDA (default: 3)
        WL: Window length (default: 2000)
        WS: Window shift (default: 1000)

    Returns:
        Array of shape (WN, 4) containing ST coefficients for each window
        where WN is the number of windows
    """
    TM = max(TAU)
    WN = int(1 + np.floor((len(data) - (WL + TM + 2 * dm - 1)) / WS))

    ST = np.full((WN, 4), np.nan)

    for wn in range(WN):
        anf = wn * WS
        ende = anf + WL + TM + 2 * dm - 1

        window_data = data[anf : ende + 1]
        ddata = deriv_all(window_data, dm)
        window_data = window_data[dm:-dm]

        STD = np.std(window_data, ddof=1)  # Use sample std to match Julia
        DATA = (window_data - np.mean(window_data)) / STD
        dDATA = ddata / STD

        # Build design matrix M with delay coordinates
        M = np.column_stack(
            [
                DATA[TM - TAU[0] : len(DATA) - TAU[0]],  # First delay coordinate
                DATA[TM - TAU[1] : len(DATA) - TAU[1]],  # Second delay coordinate
                (DATA[TM - TAU[0] : len(DATA) - TAU[0]]) ** 3,  # Nonlinear term
            ]
        )

        # Slice derivative data AFTER matrix construction
        dDATA_sliced = dDATA[TM:]

        # Solve for coefficients using normal equations
        try:
            ST[wn, :3] = np.linalg.solve(M.T @ M, M.T @ dDATA_sliced)
        except np.linalg.LinAlgError:
            ST[wn, :3] = np.linalg.lstsq(M, dDATA_sliced, rcond=None)[0]

        # Compute residual error
        ST[wn, 3] = np.sqrt(np.mean((dDATA_sliced - M @ ST[wn, :3]) ** 2))

    return ST


def compute_st_multiple(
    Y: NDArray,
    TAU: list,
    dm: int = 4,
    order: int = 3,
    WL: int = 2000,
    WS: int = 1000,
) -> NDArray:
    """
    Compute single timeseries DDA for multiple timeseries.

    Args:
        Y: Input data matrix (samples x channels)
        TAU: List of two delay values [tau1, tau2]
        dm: Derivative method parameter (default: 4)
        order: Order of DDA (default: 3)
        WL: Window length (default: 2000)
        WS: Window shift (default: 1000)

    Returns:
        Array of shape (WN, 4, n_channels) containing ST features
    """
    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)

    TM = max(TAU)
    WN = int(1 + np.floor((Y.shape[0] - (WL + TM + 2 * dm - 1)) / WS))
    n_channels = Y.shape[1]

    ST = np.full((WN, 4, n_channels), np.nan)

    for n_Y in range(n_channels):
        for wn in range(WN):
            anf = wn * WS
            ende = anf + WL + TM + 2 * dm - 1

            data = Y[anf : ende + 1, n_Y]
            ddata = deriv_all(data, dm)
            data = data[dm:-dm]

            STD = np.std(data, ddof=1)  # Use sample std to match Julia
            DATA = (data - np.mean(data)) / STD
            dDATA = ddata / STD

            # Build design matrix M with delay coordinates
            M = np.column_stack(
                [
                    DATA[TM - TAU[0] : len(DATA) - TAU[0]],
                    DATA[TM - TAU[1] : len(DATA) - TAU[1]],
                    (DATA[TM - TAU[0] : len(DATA) - TAU[0]]) ** 3,
                ]
            )

            # Slice derivative data AFTER matrix construction
            dDATA_sliced = dDATA[TM:]

            # Solve for coefficients
            try:
                ST[wn, :3, n_Y] = np.linalg.solve(M.T @ M, M.T @ dDATA_sliced)
            except np.linalg.LinAlgError:
                ST[wn, :3, n_Y] = np.linalg.lstsq(M, dDATA_sliced, rcond=None)[0]

            # Compute residual error
            ST[wn, 3, n_Y] = np.sqrt(np.mean((dDATA_sliced - M @ ST[wn, :3, n_Y]) ** 2))

    return ST


def run_dda_st_external(
    FN_DATA: str,
    FN_DDA: str,
    MODEL: NDArray,
    TAU: list,
    dm: int = 4,
    DDAorder: int = 2,
    nr_delays: int = 2,
    WL: int = 2000,
    WS: int = 1000,
    CH_list: Optional[list] = None,
    platform_system: Optional[str] = None,
) -> Tuple[str, NDArray]:
    """
    Run external DDA executable for ST analysis.

    Args:
        FN_DATA: Input data filename
        FN_DDA: Output DDA filename
        MODEL: Model specification array
        TAU: List of delay values
        dm: Derivative method parameter
        DDAorder: Order of DDA
        nr_delays: Number of delays
        WL: Window length
        WS: Window shift
        CH_list: Optional channel list
        platform_system: Platform (Windows/Unix), auto-detect if None

    Returns:
        Tuple of (command string, loaded ST results)
    """
    import platform
    import os
    import subprocess

    if platform_system is None:
        platform_system = platform.system()

    # Platform-specific executable handling
    if platform_system == "Windows":
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
    CMD += " -SELECT 1 0 0 0"  # ST only

    if CH_list:
        CMD += f" -CH_list {' '.join(map(str, CH_list))}"

    # Execute command
    if platform_system == "Windows":
        subprocess.run(CMD.split())
    else:
        subprocess.run(CMD, shell=True)

    # Load results
    ST_results = np.loadtxt(f"{FN_DDA}_ST")
    ST_results = ST_results[:, 2:]  # Skip first 2 columns

    return CMD, ST_results
