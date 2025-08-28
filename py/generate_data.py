"""
Data Generation Module for DDA Analysis

This module provides functions for generating synthetic data through ODE integration.
It supports various dynamical systems including coupled Roessler systems.
"""

import numpy as np
import platform
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List, Union
from numpy.typing import NDArray

from dda_functions import make_MOD_nr, integrate_ODE_general_BIG


def generate_roessler_data(
    n_systems: int = 4,
    a_values: Optional[Union[float, List[float]]] = None,
    b_values: Optional[Union[float, List[float]]] = None,
    c: float = 5.7,
    dt: float = 0.05,
    length: int = 20000,
    transient: int = 20000,
    delta: int = 2,
    output_file: Optional[str] = None,
    seed: Optional[int] = 42,
) -> Tuple[NDArray, dict]:
    """
    Generate data from coupled Roessler systems.

    Args:
        n_systems: Number of coupled Roessler systems
        a_values: Parameter 'a' for each system (scalar or list)
        b_values: Parameter 'b' for each system (scalar or list)
        c: Parameter 'c' (same for all systems)
        dt: Integration time step
        length: Number of integration steps
        transient: Number of transient steps to skip
        delta: Output every delta-th point
        output_file: Optional output filename
        seed: Random seed for initial conditions

    Returns:
        Tuple of (data array, parameters dictionary)
    """
    # Define Roessler system structure
    ROS = np.array(
        [
            [0, 0, 2],  # dx/dt terms
            [0, 0, 3],
            [1, 0, 1],  # dy/dt terms
            [1, 0, 2],
            [2, 0, 0],  # dz/dt terms
            [2, 0, 3],
            [2, 1, 3],
        ]
    )

    # Create model encoding
    MOD_nr, DIM, ODEorder, P = make_MOD_nr(ROS, n_systems)

    # Handle parameter values
    if a_values is None:
        a_values = [0.21] * n_systems
    elif isinstance(a_values, (float, int)):
        a_values = [a_values] * n_systems

    if b_values is None:
        b_values = [0.2 + i * 0.01 for i in range(n_systems)]
    elif isinstance(b_values, (float, int)):
        b_values = [b_values] * n_systems

    # Create parameter array
    MOD_par = []
    for i in range(n_systems):
        params = [-1, -1, 1, a_values[i], b_values[i], -c, 1]
        MOD_par.append(params)

    MOD_par = np.array(MOD_par).T.flatten()

    # Generate initial conditions
    if seed is not None:
        np.random.seed(seed)
    X0 = np.random.rand(DIM * n_systems)

    # Channel list (all coordinates)
    CH_list = list(range(1, DIM * n_systems + 1))

    # Integrate ODE
    if output_file:
        integrate_ODE_general_BIG(
            MOD_nr,
            MOD_par,
            dt,
            length,
            DIM * n_systems,
            ODEorder,
            X0,
            output_file,
            CH_list,
            delta,
            transient,
        )
        data = np.loadtxt(output_file)
    else:
        data = integrate_ODE_general_BIG(
            MOD_nr,
            MOD_par,
            dt,
            length,
            DIM * n_systems,
            ODEorder,
            X0,
            "",
            CH_list,
            delta,
            transient,
        )

    # Return data and parameters
    params = {
        "n_systems": n_systems,
        "a_values": a_values,
        "b_values": b_values,
        "c": c,
        "dt": dt,
        "length": length,
        "transient": transient,
        "delta": delta,
        "dimension": DIM,
        "order": ODEorder,
        "initial_conditions": X0,
    }

    return data, params


def generate_custom_system_data(
    system_spec: NDArray,
    n_systems: int,
    parameters: NDArray,
    dt: float = 0.05,
    length: int = 20000,
    transient: int = 20000,
    delta: int = 2,
    output_file: Optional[str] = None,
    seed: Optional[int] = 42,
) -> Tuple[NDArray, dict]:
    """
    Generate data from custom coupled ODE systems.

    Args:
        system_spec: System specification matrix (like ROS for Roessler)
        n_systems: Number of coupled systems
        parameters: Flattened parameter array
        dt: Integration time step
        length: Number of integration steps
        transient: Number of transient steps to skip
        delta: Output every delta-th point
        output_file: Optional output filename
        seed: Random seed for initial conditions

    Returns:
        Tuple of (data array, parameters dictionary)
    """
    # Create model encoding
    MOD_nr, DIM, ODEorder, P = make_MOD_nr(system_spec, n_systems)

    # Generate initial conditions
    if seed is not None:
        np.random.seed(seed)
    X0 = np.random.rand(DIM * n_systems)

    # Channel list (all coordinates)
    CH_list = list(range(1, DIM * n_systems + 1))

    # Integrate ODE
    if output_file:
        integrate_ODE_general_BIG(
            MOD_nr,
            parameters,
            dt,
            length,
            DIM * n_systems,
            ODEorder,
            X0,
            output_file,
            CH_list,
            delta,
            transient,
        )
        data = np.loadtxt(output_file)
    else:
        data = integrate_ODE_general_BIG(
            MOD_nr,
            parameters,
            dt,
            length,
            DIM * n_systems,
            ODEorder,
            X0,
            "",
            CH_list,
            delta,
            transient,
        )

    # Return data and parameters
    params = {
        "system_spec": system_spec,
        "n_systems": n_systems,
        "parameters": parameters,
        "dt": dt,
        "length": length,
        "transient": transient,
        "delta": delta,
        "dimension": DIM,
        "order": ODEorder,
        "initial_conditions": X0,
    }

    return data, params


def generate_test_data_matching_julia() -> NDArray:
    """
    Generate test data exactly matching the Julia implementation.
    This is specifically for verifying the DDA implementation.

    Returns:
        Data array matching ROS_4.ascii
    """
    # Exact parameters from run_first_DDA.jl
    NrSyst = 4

    ROS = np.array(
        [[0, 0, 2], [0, 0, 3], [1, 0, 1], [1, 0, 2], [2, 0, 0], [2, 0, 3], [2, 1, 3]]
    )

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

    # Create model encoding
    MOD_nr, DIM, ODEorder, P = make_MOD_nr(ROS, NrSyst)

    # Integration parameters
    TRANS = 20000
    dt = 0.05
    L = 20000
    DELTA = 2

    # Set random seed and generate initial conditions
    np.random.seed(42)
    X0 = np.random.rand(DIM * NrSyst)

    # Channel list (x-coordinates only: 1, 4, 7, 10)
    CH_list = list(range(1, DIM * NrSyst + 1, DIM))

    # Output filename
    FN_DATA = "ROS_4.ascii"

    # Generate data
    integrate_ODE_general_BIG(
        MOD_nr,
        MOD_par,
        dt,
        L,
        DIM * NrSyst,
        ODEorder,
        X0,
        FN_DATA,
        CH_list,
        DELTA,
        TRANS,
    )

    # Load and return data
    return np.loadtxt(FN_DATA)


def run_external_integration(
    MOD_nr: NDArray,
    MOD_par: NDArray,
    dt: float,
    L: int,
    DIM: int,
    order: int,
    X0: NDArray,
    output_file: str,
    CH_list: List[int],
    DELTA: int,
    TRANS: int = 0,
) -> str:
    """
    Run external ODE integration using command line executable.

    This function builds and executes the command line for the external
    ODE integration tool, matching the exact format used in Julia.

    Args:
        MOD_nr: Model number array
        MOD_par: Model parameters
        dt: Time step
        L: Integration length
        DIM: System dimension
        order: ODE order
        X0: Initial conditions
        output_file: Output filename
        CH_list: Channel list
        DELTA: Output sampling
        TRANS: Transient steps

    Returns:
        Command string that was executed
    """
    # Platform-specific executable
    if platform.system() == "Windows":
        if not Path("i_ODE_general_BIG.exe").exists():
            import shutil

            shutil.copy("i_ODE_general_BIG", "i_ODE_general_BIG.exe")
        CMD = ".\\i_ODE_general_BIG.exe"
    else:
        CMD = "./i_ODE_general_BIG"

    # Build command matching Julia exactly
    MOD_NR = " ".join(map(str, MOD_nr))
    CMD = f"{CMD} -MODEL {MOD_NR}"

    MOD_PAR = " ".join(map(str, MOD_par))
    CMD = f"{CMD} -PAR {MOD_PAR}"

    ANF = " ".join(map(str, X0))
    CMD = f"{CMD} -ANF {ANF}"

    CMD = f"{CMD} -dt {dt}"
    CMD = f"{CMD} -L {L}"
    CMD = f"{CMD} -DIM {DIM}"
    CMD = f"{CMD} -order {order}"

    if TRANS > 0:
        CMD = f"{CMD} -TRANS {TRANS}"

    CMD = f"{CMD} -FILE {output_file}"
    CMD = f"{CMD} -DELTA {DELTA}"
    CMD = f"{CMD} -CH_list {' '.join(map(str, CH_list))}"

    # Execute command
    if platform.system() == "Windows":
        subprocess.run(CMD.split())
    else:
        subprocess.run(CMD, shell=True)

    return CMD


if __name__ == "__main__":
    # Example: Generate Roessler data
    print("Generating coupled Roessler systems data...")
    data, params = generate_roessler_data(
        n_systems=4, output_file="test_roessler.ascii"
    )
    print(f"Generated data shape: {data.shape}")
    print(f"Parameters: {params}")

    # Example: Generate test data matching Julia
    print("\nGenerating test data matching Julia implementation...")
    test_data = generate_test_data_matching_julia()
    print(f"Test data shape: {test_data.shape}")
