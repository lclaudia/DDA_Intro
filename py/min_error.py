"""
MinError module for DDA (Delay Differential Analysis).

Python equivalent of MinError.jl for finding optimal models and delays.
"""

import os
import platform
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Union
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from dda_functions import monomial_list, make_MODEL, SL

# Global variable for number of delays
nr_delays = 2


def make_MODEL_new(MOD: NDArray, SSYM: NDArray, mm: int) -> Tuple[NDArray, NDArray, str, int]:
    """
    Create model from MOD array.
    
    Args:
        MOD: Model specification array
        SSYM: Symmetry array
        mm: Model index
    
    Returns:
        Tuple of (MODEL indices, SYM, model string, L_AF)
    """
    MODEL = np.where(MOD[mm, :] == 1)[0] + 1  # 1-based indexing for compatibility
    L_AF = len(MODEL) + 1
    SYM = SSYM[mm, :]
    model = "_".join([f"{x:02d}" for x in MODEL])
    
    return MODEL, SYM, model, L_AF


def make_MOD_new_new(N_MOD: Union[int, List[int]], nr_delays: int, order: int) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Generate models and symmetry information.
    
    Args:
        N_MOD: Number of model terms (or list of numbers)
        nr_delays: Number of delays (must be 2)
        order: Order of the DDA
    
    Returns:
        Tuple of (MOD array, P_DDA monomial list, SSYM symmetry array)
    """
    if nr_delays != 2:
        print("only nr_delays=2 supported")
        nr_delays = 2
    
    P_DDA = monomial_list(nr_delays, order)
    L = len(P_DDA)
    
    # Convert P_DDA for symmetry checking
    PP = -P_DDA.copy()
    PP[PP == -1] = 2
    PP[PP == -2] = 1
    PP = np.sort(PP, axis=1)
    
    # Find symmetric pairs
    f = np.zeros((len(P_DDA), 2), dtype=int)
    for k1 in range(len(P_DDA)):
        f[k1, 0] = k1
        for k2 in range(len(P_DDA)):
            if np.sum(np.abs(P_DDA - PP[k1, :])) == 0:
                f[k1, 1] = k2
                break
    
    # Ensure N_MOD is a list
    if isinstance(N_MOD, int):
        N_MOD = [N_MOD]
    
    MOD_list = []
    
    for N in N_MOD:
        # Generate all combinations
        C = list(combinations(range(L), N))
        M = np.zeros((len(C), L), dtype=int)
        
        for c_idx, c in enumerate(C):
            M[c_idx, list(c)] = 1
        
        # Find symmetric models
        M1 = np.sort(M * np.arange(1, L + 1), axis=1)[:, -N:]
        M2 = -M1.copy()
        
        for k1 in range(len(f)):
            M2[M2 == -f[k1, 0]] = f[k1, 1]
        
        M2 = np.sort(M2, axis=1)
        
        # Find unique non-symmetric models
        f2 = np.zeros((len(M1), 2), dtype=int)
        for k1 in range(len(M1)):
            f2[k1, 0] = k1
            for k2 in range(len(M1)):
                if np.array_equal(M1[k1, :], M2[k2, :]):
                    f2[k1, 1] = k2
                    break
        
        f2 = np.sort(f2, axis=1)
        f2 = np.unique(f2, axis=0)
        f2 = f2[f2[:, 0] != f2[:, 1], 1]
        f2_set = set(f2)
        keep_indices = [i for i in range(len(M1)) if i not in f2_set]
        
        if keep_indices:
            MOD_list.append(M[keep_indices, :])
    
    if MOD_list:
        MOD = np.vstack(MOD_list)
    else:
        MOD = np.zeros((0, L), dtype=int)
    
    # Calculate symmetry information
    SSYM = np.full((len(MOD), 2), -1, dtype=int)
    for n_M in range(len(MOD)):
        p = P_DDA[np.where(MOD[n_M, :] == 1)[0], :]
        
        # Number of unique non-zero delays
        SSYM[n_M, 0] = len(np.unique(p[p > 0]))
        
        # Check for symmetry
        p_float = p.astype(float)
        p_float[p_float == 0] = np.nan
        p1 = ((p_float + 2) % 2) + 1
        p1[np.isnan(p1)] = 0
        p1 = p1.astype(int)
        p_float[np.isnan(p_float)] = 0
        p = p_float.astype(int)
        
        p1 = np.sort(p1, axis=1)
        p1 = p1[np.lexsort(p1.T)]
        
        if np.array_equal(p, p1):
            SSYM[n_M, 1] = 1
        else:
            SSYM[n_M, 1] = 0
    
    return MOD, P_DDA, SSYM


def make_TAU_ALL(SSYM: NDArray, DELAYS: NDArray) -> None:
    """
    Generate all possible delay combinations based on symmetry.
    
    Args:
        SSYM: Symmetry array
        DELAYS: Array of possible delays
    """
    uSYM = np.unique(SSYM, axis=0)
    
    for k in range(len(uSYM)):
        nr = uSYM[k, 0]
        sym = uSYM[k, 1]
        
        FN = f"TAU_ALL__{nr}_{sym}"
        
        with open(FN, 'w') as fid:
            if nr == 1:
                for tau1 in range(len(DELAYS)):
                    fid.write(f"{DELAYS[tau1]}\n")
            elif nr == 2:
                if sym == 0:
                    for tau1 in range(len(DELAYS)):
                        for tau2 in range(len(DELAYS)):
                            if tau1 != tau2:
                                fid.write(f"{DELAYS[tau1]} {DELAYS[tau2]}\n")
                elif sym == 1:
                    for tau1 in range(len(DELAYS)):
                        for tau2 in range(len(DELAYS)):
                            if tau1 < tau2:
                                fid.write(f"{DELAYS[tau1]} {DELAYS[tau2]}\n")


def MinError(
    FN_data: str,
    CH_list: Union[List, NDArray],
    DDA_DIR: str,
    DDAorder: int,
    dm: int,
    WL: Union[int, List],
    WS: Union[int, List],
    DELAYS: NDArray,
    yn: str,
    ALLE: str,
    StartEnd: List,
    AsciiEdf: str,
    DDAmodel: Optional[NDArray] = None
) -> Tuple[Optional[NDArray], Optional[NDArray]]:
    """
    Find optimal model and delays for DDA analysis.
    
    Args:
        FN_data: Input data filename
        CH_list: List of channel groups to analyze
        DDA_DIR: Output directory for DDA results
        DDAorder: Order of DDA
        dm: Derivative method parameter
        WL: Window length (empty list for no windowing)
        WS: Window shift (empty list for no windowing)
        DELAYS: Array of possible delays
        yn: "yes" to recompute, otherwise use existing
        ALLE: "ALL" to use all channels
        StartEnd: Start and end indices for data
        AsciiEdf: "ASCII" or "EDF" data format
        DDAmodel: Optional specific model to use
    
    Returns:
        Tuple of (TAU_select, mm_select) - selected delays and model indices
    """
    global nr_delays
    
    STOP = 0
    
    if DDAmodel is not None and len(DDAmodel) > 0:
        MODEL, L_AF, DDAorder1 = make_MODEL(DDAmodel)
        if DDAorder1 != DDAorder:
            STOP = 1
        
        # Calculate symmetry
        unique_vals = np.unique(DDAmodel[DDAmodel > 0])
        SYM = [len(unique_vals), -1]
        
        DDAmodel2 = DDAmodel.copy()
        DDAmodel2[DDAmodel2 == 1] = -1
        DDAmodel2[DDAmodel2 == 2] = 1
        DDAmodel2[DDAmodel2 == -1] = 2
        DDAmodel2 = np.sort(DDAmodel2, axis=1)
        DDAmodel2 = DDAmodel2[np.lexsort(DDAmodel2.T)]
        
        if np.array_equal(DDAmodel, DDAmodel2):
            SYM[1] = 1
        else:
            SYM[1] = 0
        
        SSYM = np.array([SYM])
        P_DDA = monomial_list(nr_delays, DDAorder)
        MOD = np.zeros((1, len(P_DDA)), dtype=int)
        MOD[0, MODEL - 1] = 1  # Convert to 0-based indexing
        model = "_".join([f"{x:02d}" for x in MODEL])
        N_MOD = DDAmodel.shape[1]
    else:
        DDAmodel = []
    
    # Process channel list
    if isinstance(CH_list, range):
        CH_list = list(CH_list)
    
    NrCH = len(CH_list) if CH_list else 0
    
    if NrCH > 0:
        # Flatten channel list
        if isinstance(CH_list[0], (list, np.ndarray)):
            CHs = []
            for ch_group in CH_list:
                if isinstance(ch_group, (list, np.ndarray)):
                    CHs.extend(ch_group)
                else:
                    CHs.append(ch_group)
        else:
            CHs = CH_list
        
        if ALLE != "ALL":
            CHs_su = sorted(set(CHs))
            ChIDX = [CHs_su.index(ch) for ch in CHs]
    else:
        ALLE = ""
    
    if len(DDAmodel) == 0:
        N_MOD = 3
        MOD, P_DDA, SSYM = make_MOD_new_new(N_MOD, nr_delays, DDAorder)
    
    if STOP == 0:
        make_TAU_ALL(SSYM, DELAYS)
        
        MIN = np.full(NrCH, 1000.0)
        mm_select = np.zeros(NrCH, dtype=int)
        TAU_select = np.full((NrCH, 2), -1, dtype=int)
        
        for mm in range(len(MOD)):
            if len(DDAmodel) == 0:
                MODEL, SYM, model, L_AF = make_MODEL_new(MOD, SSYM, mm)
            
            TAU_name = f"TAU_ALL__{SYM[0]}_{SYM[1]}"
            TAU = np.loadtxt(TAU_name, ndmin=2)
            N_TAU = len(TAU)
            
            FN_DDA = f"{DDA_DIR}{SL}{model}"
            
            if not os.path.isfile(f"{FN_DDA}_ST") or yn == "yes":
                print(mm)
                
                # Prepare executable
                if platform.system() == "Windows":
                    if not Path("run_DDA_AsciiEdf.exe").exists():
                        subprocess.run(["cp", "run_DDA_AsciiEdf", "run_DDA_AsciiEdf.exe"])
                    CMD = ".\\run_DDA_AsciiEdf.exe"
                else:
                    CMD = "./run_DDA_AsciiEdf"
                
                # Build command
                if AsciiEdf == "ASCII":
                    CMD = f"{CMD} -ASCII"
                else:
                    CMD = f"{CMD} -EDF"
                
                CMD = f"{CMD} -MODEL {' '.join(map(str, MODEL))}"
                CMD = f"{CMD} -TAU_file {TAU_name}"
                CMD = f"{CMD} -dm {dm} -order {DDAorder} -nr_tau {nr_delays}"
                CMD = f"{CMD} -DATA_FN {FN_data} -OUT_FN {FN_DDA}"
                CMD = f"{CMD} -SELECT 1 0 0 0"
                
                if len(StartEnd) > 0:
                    CMD = f"{CMD} -StartEnd {' '.join(map(str, StartEnd))}"
                
                if NrCH > 0 and ALLE != "ALL":
                    CMD = f"{CMD} -CH_list {' '.join(map(str, CHs_su))}"
                
                if WL and len(WL) > 0:
                    CMD = f"{CMD} -WL {WL}"
                
                if WS and len(WS) > 0:
                    CMD = f"{CMD} -WS {WS}"
                
                # Execute command
                if platform.system() == "Windows":
                    subprocess.run(CMD.split())
                else:
                    subprocess.run(CMD, shell=True)
            
            # Process results
            if NrCH > 0:
                AF = np.loadtxt(f"{FN_DDA}_ST")
                AF = AF[:, 2:]  # Skip first two columns
                WN = len(AF)
                
                if ALLE == "ALL":
                    CHs_su = list(range(1, int(AF.shape[1] / L_AF / N_TAU) + 1))
                    ChIDX = [CHs_su.index(ch - 1) for ch in CHs]  # Adjust for 0-based
                
                # Reshape and process
                AF = AF.reshape(WN, L_AF, len(CHs_su), N_TAU)[:, -1, :, :]
                AF = AF[:, ChIDX, :]
                
                # Process channel groups
                L_CH = [0] + [len(ch) if isinstance(ch, (list, np.ndarray)) else 1 
                             for ch in CH_list]
                L_CH = np.cumsum(L_CH)
                
                for k in range(NrCH):
                    AF[0, k, :] = np.mean(AF[:, L_CH[k]:L_CH[k+1], :], axis=1)
                
                AF = np.mean(AF[:, :NrCH, :], axis=0)
                
                # Find minimum
                M = np.min(AF, axis=1)
                for k in range(NrCH):
                    if M[k] < MIN[k]:
                        MIN[k] = M[k]
                        mm_select[k] = mm
                        min_idx = np.where(AF[k, :] == M[k])[0][0]
                        TAU_select[k, :TAU.shape[1]] = TAU[min_idx, :]
        
        # Print results
        if NrCH > 0:
            for k in range(NrCH):
                if isinstance(CH_list[k], (list, np.ndarray)):
                    ch_vals = CH_list[k]
                else:
                    ch_vals = [CH_list[k]]
                
                print(f"\nCH = {' '.join(map(str, ch_vals))} :")
                print("  Best model:")
                
                p = P_DDA[np.where(MOD[mm_select[k], :] == 1)[0], :]
                for i in range(N_MOD):
                    print(f"    [ {' '.join(map(str, p[i, :]))} ]")
                
                print("  Best delays: [", end="")
                for i in range(nr_delays):
                    if TAU_select[k, i] > 0:
                        print(f" {TAU_select[k, i]}", end="")
                print(" ]\n")
        
        return TAU_select, mm_select
    
    else:
        print("\nDDA order inconsistent\n")
        return None, None