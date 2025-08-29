"""
Cross Timeseries DDA Analysis Module

This module provides functions for performing DDA (Delay Differential Analysis)
on pairs of timeseries data.
"""

import numpy as np
from typing import Tuple, Optional, List
from numpy.typing import NDArray
from itertools import combinations
from pathlib import Path

from dda_functions import deriv_all


def compute_ct_pair(
    data1: NDArray,
    data2: NDArray,
    TAU: list,
    dm: int = 4,
    order: int = 3,
    WL: int = 2000,
    WS: int = 1000,
) -> NDArray:
    """
    Compute cross-timeseries DDA outputs for a pair of signals.

    Args:
        data1: First timeseries data (1D array)
        data2: Second timeseries data (1D array)
        TAU: List of two delay values [tau1, tau2]
        dm: Derivative method parameter (default: 4)
        order: Order of DDA (default: 3)
        WL: Window length (default: 2000)
        WS: Window shift (default: 1000)

    Returns:
        Array of shape (WN, 4) containing CT coefficients for each window
    """
    TM = max(TAU)
    WN = int(1 + np.floor((len(data1) - (WL + TM + 2 * dm - 1)) / WS))

    CT = np.full((WN, 4), np.nan)

    for wn in range(WN):
        anf = wn * WS
        ende = anf + WL + TM + 2 * dm - 1

        # Process first timeseries
        window_data1 = data1[anf : ende + 1]
        ddata1 = deriv_all(window_data1, dm)
        window_data1 = window_data1[dm:-dm]

        STD1 = np.std(window_data1, ddof=1)
        DATA1 = (window_data1 - np.mean(window_data1)) / STD1
        dDATA1 = ddata1 / STD1

        # Process second timeseries
        window_data2 = data2[anf : ende + 1]
        ddata2 = deriv_all(window_data2, dm)
        window_data2 = window_data2[dm:-dm]

        STD2 = np.std(window_data2, ddof=1)
        DATA2 = (window_data2 - np.mean(window_data2)) / STD2
        dDATA2 = ddata2 / STD2

        # Build design matrices
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

        # Solve for coefficients
        try:
            CT[wn, :3] = np.linalg.solve(M.T @ M, M.T @ dDATA_combined)
        except np.linalg.LinAlgError:
            CT[wn, :3] = np.linalg.lstsq(M, dDATA_combined, rcond=None)[0]

        # Compute residual error
        CT[wn, 3] = np.sqrt(np.mean((dDATA_combined - M @ CT[wn, :3]) ** 2))

    return CT


def compute_ct_multiple(
    Y: NDArray,
    TAU: list,
    dm: int = 4,
    order: int = 3,
    WL: int = 2000,
    WS: int = 1000,
    channel_pairs: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[NDArray, NDArray]:
    """
    Compute cross-timeseries DDA for multiple channel pairs.

    Args:
        Y: Input data matrix (samples x channels)
        TAU: List of two delay values [tau1, tau2]
        dm: Derivative method parameter (default: 4)
        order: Order of DDA (default: 3)
        WL: Window length (default: 2000)
        WS: Window shift (default: 1000)
        channel_pairs: Optional list of (ch1, ch2) pairs (0-based).
                      If None, all combinations are computed.

    Returns:
        Tuple of (CT array, LIST of channel pairs)
        CT has shape (WN, 4, n_pairs)
        LIST has shape (n_pairs, 2) with 0-based channel indices
    """
    NrCH = Y.shape[1]

    if channel_pairs is None:
        # Generate all combinations
        CH = list(range(NrCH))
        LIST = list(combinations(CH, 2))
        LIST = np.array(LIST)
    else:
        LIST = np.array(channel_pairs)

    TM = max(TAU)
    WN = int(1 + np.floor((Y.shape[0] - (WL + TM + 2 * dm - 1)) / WS))

    CT = np.full((WN, 4, len(LIST)), np.nan)

    for n_LIST in range(len(LIST)):
        ch1, ch2 = LIST[n_LIST]

        for wn in range(WN):
            anf = wn * WS
            ende = anf + WL + TM + 2 * dm - 1

            # Process first channel
            data1 = Y[anf : ende + 1, ch1]
            ddata1 = deriv_all(data1, dm)
            data1 = data1[dm:-dm]

            STD = np.std(data1, ddof=1)
            DATA1 = (data1 - np.mean(data1)) / STD
            dDATA1 = ddata1 / STD

            # Process second channel
            data2 = Y[anf : ende + 1, ch2]
            ddata2 = deriv_all(data2, dm)
            data2 = data2[dm:-dm]

            STD = np.std(data2, ddof=1)
            DATA2 = (data2 - np.mean(data2)) / STD
            dDATA2 = ddata2 / STD

            # Build design matrices
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

            # Solve for coefficients
            try:
                CT[wn, :3, n_LIST] = np.linalg.solve(M.T @ M, M.T @ dDATA_combined)
            except np.linalg.LinAlgError:
                CT[wn, :3, n_LIST] = np.linalg.lstsq(M, dDATA_combined, rcond=None)[0]

            # Compute residual error
            CT[wn, 3, n_LIST] = np.sqrt(
                np.mean((dDATA_combined - M @ CT[wn, :3, n_LIST]) ** 2)
            )

    return CT, LIST


def run_dda_ct_external(
    FN_DATA: str,
    FN_DDA: str,
    MODEL: NDArray,
    TAU: list,
    LIST: NDArray,
    dm: int = 4,
    DDAorder: int = 2,
    nr_delays: int = 2,
    WL: int = 2000,
    WS: int = 1000,
    WL_CT: int = 2,
    WS_CT: int = 2,
    platform_system: Optional[str] = None,
) -> Tuple[str, NDArray]:
    """
    Run external DDA executable for CT analysis.

    Args:
        FN_DATA: Input data filename
        FN_DDA: Output DDA filename
        MODEL: Model specification array
        TAU: List of delay values
        LIST: Array of channel pairs (0-based)
        dm: Derivative method parameter
        DDAorder: Order of DDA
        nr_delays: Number of delays
        WL: Window length
        WS: Window shift
        WL_CT: CT-specific window length multiplier
        WS_CT: CT-specific window shift multiplier
        platform_system: Platform (Windows/Unix), auto-detect if None

    Returns:
        Tuple of (command string, loaded CT results)
    """
    import platform
    import subprocess

    if platform_system is None:
        platform_system = platform.system()

    # Platform-specific executable handling
    if platform_system == "Windows":
        executable = Path("run_DDA_AsciiEdf.exe")
        if not executable.exists():
            import shutil

            shutil.copy("run_DDA_AsciiEdf", str(executable))
        CMD = str(executable)
    else:
        CMD = str(Path("run_DDA_AsciiEdf"))

    # Build command
    CMD += " -ASCII"
    CMD += f" -MODEL {' '.join(map(str, MODEL))}"
    CMD += f" -TAU {' '.join(map(str, TAU))}"
    CMD += f" -dm {dm} -order {DDAorder} -nr_tau {nr_delays}"
    CMD += f" -DATA_FN {FN_DATA} -OUT_FN {FN_DDA}"
    CMD += f" -WL {WL} -WS {WS}"
    CMD += " -SELECT 0 1 0 0"  # CT only

    # Convert 0-based LIST to 1-based for external tool
    LIST_1based = LIST + 1
    CMD += f" -CH_list {' '.join(map(str, LIST_1based.flatten()))}"
    CMD += f" -WL_CT {WL_CT} -WS_CT {WS_CT}"

    # Execute command
    if platform_system == "Windows":
        subprocess.run(CMD.split())
    else:
        subprocess.run(CMD, shell=True)

    # Load results
    CT_results = np.loadtxt(f"{FN_DDA}_CT")
    CT_results = CT_results[:, 2:]  # Skip first 2 columns

    return CMD, CT_results
