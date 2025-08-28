"""
Structure Selection module for DDA (Delay Differential Analysis).

Python equivalent of StructureSelection.jl for analyzing coupled Roessler systems.
"""

import os
import numpy as np

from dda_functions import make_MOD_nr, integrate_ODE_general_BIG, dir_exist, SL
from min_error import MinError


def main():
    """
    Main function for structure selection analysis of coupled Roessler systems.
    """

    # Define Roessler system structure
    ROS = np.array(
        [[0, 0, 2], [0, 0, 3], [1, 0, 1], [1, 0, 2], [2, 0, 0], [2, 0, 3], [2, 1, 3]]
    )

    # Number of coupled systems
    NrSyst = 10

    # Create model encoding
    MOD_nr, DIM, ODEorder, P = make_MOD_nr(ROS, NrSyst)

    # Example identifier
    example = "_1"

    # System parameters
    a = 0.15
    b = 0.2
    c = 10

    # Create parameter array for all systems
    base_params = np.array([-1, -1, 1, a, b, -c, 1])
    MOD_par = np.tile(base_params, (NrSyst, 1))

    # Add slight variations to b parameter
    MOD_par[:, 4] += np.random.randn(NrSyst) / 100

    # Flatten parameter array
    MOD_par = MOD_par.T.flatten()

    # Integration parameters
    LL = 10000  # Integration length
    TRANS = 20000  # Transient steps
    dt = 0.05  # Integration step size
    X0 = np.random.rand(DIM * NrSyst)  # Initial conditions

    # Data directory and filename
    DATA_DIR = "DATA"
    dir_exist(DATA_DIR)
    FN_data = f"{DATA_DIR}{SL}data_{example}.ascii"

    # Channel list (all x, y, z coordinates)
    CH_list_all = list(range(1, DIM * NrSyst + 1))
    DELTA = 2  # Take every second data point

    # Generate data if it doesn't exist
    if not os.path.isfile(FN_data):
        integrate_ODE_general_BIG(
            MOD_nr,
            MOD_par,
            dt,
            LL,
            DIM * NrSyst,
            ODEorder,
            X0,
            FN_data,
            CH_list_all,
            DELTA,
            TRANS,
        )

    # DDA directory
    DDA_DIR = f"DDA_{example}"
    dir_exist(DDA_DIR)

    # DDA parameters
    dm = 4
    TM = 50
    DELAYS = np.arange(dm + 1, TM + 1)

    # Window parameters (empty for no windowing)
    WL = []
    WS = []

    # DDA order
    DDAorder = 2

    # Run MinError for all timeseries
    print("Computing DDA outputs for all timeseries...")
    CH_list = []
    MinError(
        FN_data,
        CH_list,
        DDA_DIR,
        DDAorder,
        dm,
        WL,
        WS,
        DELAYS,
        "yes",
        "ALL",
        [],
        "ASCII",
    )

    # Analyze only x components (common best model)
    print("\nAnalyzing x components (indices 1, 4, 7, ...):")
    CH_list = [list(range(1, DIM * NrSyst + 1, 3))]  # 1, 4, 7, 10, ...
    TAU_select_x, mm_select_x = MinError(
        FN_data, CH_list, DDA_DIR, DDAorder, dm, WL, WS, DELAYS, "", "ALL", [], "ASCII"
    )

    # Analyze only y components (common best model)
    print("\nAnalyzing y components (indices 2, 5, 8, ...):")
    CH_list = [list(range(2, DIM * NrSyst + 1, 3))]  # 2, 5, 8, 11, ...
    TAU_select_y, mm_select_y = MinError(
        FN_data, CH_list, DDA_DIR, DDAorder, dm, WL, WS, DELAYS, "", "ALL", [], "ASCII"
    )

    # Analyze each timeseries individually
    print("\nAnalyzing each timeseries individually:")
    CH_list = [[i] for i in range(1, DIM * NrSyst + 1)]
    TAU_select_ind, mm_select_ind = MinError(
        FN_data, CH_list, DDA_DIR, DDAorder, dm, WL, WS, DELAYS, "", "ALL", [], "ASCII"
    )

    # Analyze specific combinations
    print("\nAnalyzing specific timeseries combinations:")
    CH_list = [
        [4, 7, 10],  # Specific x components
        [19],  # Single component
        [25, 22],  # Two specific components
    ]
    TAU_select_combo, mm_select_combo = MinError(
        FN_data, CH_list, DDA_DIR, DDAorder, dm, WL, WS, DELAYS, "", "ALL", [], "ASCII"
    )

    # Partial data analysis
    print("\n" + "=" * 60)
    print("Analyzing partial data (samples 1-2000):")
    DDA_DIR_part = f"DDA_{example}_part"
    dir_exist(DDA_DIR_part)

    CH_list = [[4, 7, 10], [19], [25, 22]]
    TAU_select_part, mm_select_part = MinError(
        FN_data,
        CH_list,
        DDA_DIR_part,
        DDAorder,
        dm,
        WL,
        WS,
        DELAYS,
        "yes",
        "ALL",
        [1, 2000],
        "ASCII",
    )

    # Return results
    return {
        "x_components": (TAU_select_x, mm_select_x),
        "y_components": (TAU_select_y, mm_select_y),
        "individual": (TAU_select_ind, mm_select_ind),
        "combinations": (TAU_select_combo, mm_select_combo),
        "partial": (TAU_select_part, mm_select_part),
    }


if __name__ == "__main__":
    results = main()
    print("\nStructure selection analysis complete!")
    print(f"Results stored in dictionary with keys: {list(results.keys())}")
