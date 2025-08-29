#!/usr/bin/env python3
"""
Test DDA Functions Module

This module verifies that the refactored DDA functions produce the same results
as the original run_first_DDA.py implementation, with errors on the order of e-17.
"""

import numpy as np
import os
from typing import Tuple

from dda_st import compute_st_single, compute_st_multiple
from dda_ct import compute_ct_multiple
from dda_de import compute_dynamical_ergodicity, plot_ergodicity_heatmap


def verify_dda_accuracy() -> Tuple[float, float, bool]:
    """
    Verify that the DDA functions produce results matching the external DDA tool
    with errors on the order of e-17.

    Returns:
        Tuple of (ST error, CT error, success flag)
    """
    # Check for required data file
    FN_DATA = "ROS_4.ascii"
    if not os.path.isfile(FN_DATA):
        print(
            f"Error: {FN_DATA} file not found. Please run Julia version first to generate data."
        )
        return np.nan, np.nan, False

    # Load data
    Y = np.loadtxt(FN_DATA)
    print(f"Loaded data shape: {Y.shape}")

    # DDA parameters (matching original implementation)
    TAU = [32, 9]
    dm = 4
    order = 3
    WL = 2000
    WS = 1000

    # Test single timeseries DDA
    print("\n" + "=" * 60)
    print("Testing Single Timeseries DDA...")
    Y_single = Y[:, 0]
    ST_single_dict = compute_st_single(Y_single, TAU, dm, order, WL, WS, return_dict=True)
    ST_single = compute_st_single(Y_single, TAU, dm, order, WL, WS, return_dict=False)
    print(f"ST_single shape: {ST_single.shape}")
    print(f"Dictionary keys: {list(ST_single_dict.keys())}")

    # Test multiple timeseries DDA
    print("\n" + "=" * 60)
    print("Testing Multiple Timeseries DDA...")
    ST_multiple_dict = compute_st_multiple(Y, TAU, dm, order, WL, WS, return_dict=True)
    ST_multiple = compute_st_multiple(Y, TAU, dm, order, WL, WS, return_dict=False)
    print(f"ST_multiple shape: {ST_multiple.shape}")
    print(f"Dictionary keys: {list(ST_multiple_dict.keys())}")
    print(f"Metadata: {ST_multiple_dict['metadata']}")

    # Test cross-timeseries DDA
    print("\n" + "=" * 60)
    print("Testing Cross-Timeseries DDA...")
    CT, channel_pairs = compute_ct_multiple(Y, TAU, dm, order, WL, WS)
    print(f"CT shape: {CT.shape}")
    print(f"Number of channel pairs: {len(channel_pairs)}")

    # Test dynamical ergodicity
    print("\n" + "=" * 60)
    print("Testing Dynamical Ergodicity...")
    E = compute_dynamical_ergodicity(ST_multiple, CT, channel_pairs)
    print(f"Ergodicity matrix shape: {E.shape}")

    # Load external DDA results for comparison
    ST_external_file = "ROS_4.DDA_ST"
    CT_external_file = "ROS_4.DDA_CT"

    if os.path.isfile(ST_external_file) and os.path.isfile(CT_external_file):
        print("\n" + "=" * 60)
        print("Comparing with external DDA results...")

        # Load external results
        ST_external = np.loadtxt(ST_external_file)
        ST_external = ST_external[:, 2:]  # Skip first 2 columns

        CT_external = np.loadtxt(CT_external_file)
        CT_external = CT_external[:, 2:]  # Skip first 2 columns

        # Reshape our results to match external format
        # Julia uses column-major reshaping, Python uses row-major by default
        WN = ST_multiple.shape[0]
        ST_reshaped = ST_multiple.reshape(WN, -1, order="F")
        CT_reshaped = CT.reshape(WN, -1, order="F")

        # Compute errors using absolute maximum difference
        diff_ST = np.abs(ST_reshaped - ST_external)
        diff_CT = np.abs(CT_reshaped - CT_external)
        
        error_ST_max = np.max(diff_ST)
        error_ST_mean = np.mean(diff_ST)
        error_CT_max = np.max(diff_CT)
        error_CT_mean = np.mean(diff_CT)

        print(f"\nST max error: {error_ST_max:.2e}")
        print(f"ST mean error: {error_ST_mean:.2e}")
        print(f"CT max error: {error_CT_max:.2e}")
        print(f"CT mean error: {error_CT_mean:.2e}")

        # Check if errors are within machine precision
        tolerance = 1e-17
        st_within_tolerance = error_ST_max < tolerance
        ct_within_tolerance = error_CT_max < tolerance

        print(f"\nST within tolerance (< {tolerance:.0e}): {st_within_tolerance}")
        print(f"CT within tolerance (< {tolerance:.0e}): {ct_within_tolerance}")

        success = st_within_tolerance and ct_within_tolerance

        if success:
            print(
                "\n✓ SUCCESS: All errors are within expected tolerance (order of e-17)"
            )
        else:
            print("\n✗ FAILURE: Errors exceed expected tolerance")

        return error_ST_max, error_CT_max, success

    else:
        print("\nWarning: External DDA result files not found.")
        print(f"  Missing: {ST_external_file} and/or {CT_external_file}")
        print("  Cannot perform accuracy verification.")
        return np.nan, np.nan, False


def test_individual_functions():
    """
    Test individual DDA functions with simple data.
    """
    print("\n" + "=" * 60)
    print("Testing Individual Functions with Simple Data")
    print("=" * 60)

    # Create simple test data
    t = np.linspace(0, 100, 5000)
    test_data = np.column_stack(
        [
            np.sin(0.1 * t) + 0.1 * np.random.randn(len(t)),
            np.cos(0.1 * t) + 0.1 * np.random.randn(len(t)),
            np.sin(0.2 * t) + 0.1 * np.random.randn(len(t)),
        ]
    )

    TAU = [10, 5]
    dm = 4

    # Test ST single
    print("\nTesting compute_st_single...")
    st_result_dict = compute_st_single(test_data[:, 0], TAU, dm, return_dict=True)
    st_result = compute_st_single(test_data[:, 0], TAU, dm, return_dict=False)
    print(f"  Array shape: {st_result.shape}")
    print(f"  Dictionary coefficients shape: {st_result_dict['coefficients'].shape}")
    print(f"  Mean error: {np.mean(st_result[:, 3]):.6f}")
    print(f"  Model equation: {st_result_dict['model_description']['equation']}")

    # Test ST multiple
    print("\nTesting compute_st_multiple...")
    st_multi_dict = compute_st_multiple(test_data, TAU, dm, return_dict=True, 
                                        channel_names=['sin1', 'cos1', 'sin2'])
    st_multi_result = compute_st_multiple(test_data, TAU, dm, return_dict=False)
    print(f"  Array shape: {st_multi_result.shape}")
    print(f"  Dictionary coefficients shape: {st_multi_dict['coefficients'].shape}")
    print(f"  Channel names: {st_multi_dict['metadata'].get('channel_names')}")
    print(f"  Mean errors per channel: {np.mean(st_multi_result[:, 3, :], axis=0)}")

    # Test CT
    print("\nTesting compute_ct_multiple...")
    ct_result, pairs = compute_ct_multiple(test_data, TAU, dm)
    print(f"  Result shape: {ct_result.shape}")
    print(f"  Channel pairs: {pairs}")
    print(f"  Mean errors per pair: {np.mean(ct_result[:, 3, :], axis=0)}")

    # Test DE
    print("\nTesting compute_dynamical_ergodicity...")
    E = compute_dynamical_ergodicity(st_multi_result, ct_result, pairs)
    print(f"  Ergodicity matrix shape: {E.shape}")
    print(f"  Ergodicity values:\n{E}")


def main():
    """
    Main test function.
    """
    print("DDA Function Test Suite")
    print("=" * 60)

    # Test with simple data first
    test_individual_functions()

    # Verify accuracy against external results
    error_ST_max, error_CT_max, success = verify_dda_accuracy()

    # Create ergodicity plot
    if success:
        print("\n" + "=" * 60)
        print("Creating Ergodicity Heatmap...")

        # Reload data and compute ergodicity
        Y = np.loadtxt("ROS_4.ascii")
        TAU = [32, 9]

        ST = compute_st_multiple(Y, TAU, return_dict=False)
        CT, pairs = compute_ct_multiple(Y, TAU)
        E = compute_dynamical_ergodicity(ST, CT, pairs)

        plot_ergodicity_heatmap(
            E,
            title="Dynamical Ergodicity - Verified Implementation",
            save_path="ergodicity_heatmap_verified.png",
        )
        print("Heatmap saved as: ergodicity_heatmap_verified.png")

    print("\n" + "=" * 60)
    print("Test suite complete.")

    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
