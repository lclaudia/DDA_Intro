#!/usr/bin/env python3
"""
High-Precision DDA Accuracy Test

This module ensures that the Python DDA implementation matches the external
binary results within machine precision (1e-17).
"""

import numpy as np
import os
import sys

from dda_st import compute_st_single, compute_st_multiple, run_dda_st_external
from dda_ct import compute_ct_multiple, run_dda_ct_external


def compare_arrays(
    arr1: np.ndarray, arr2: np.ndarray, name: str, tolerance: float = 1e-17
) -> bool:
    """
    Compare two arrays and report detailed error statistics.

    Args:
        arr1: First array
        arr2: Second array
        name: Name for reporting
        tolerance: Maximum allowed error

    Returns:
        True if arrays match within tolerance
    """
    diff = np.abs(arr1 - arr2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)

    # Find location of maximum difference
    max_idx = np.unravel_index(np.argmax(diff), diff.shape)

    print(f"\n{name} Comparison:")
    print(f"  Shape: {arr1.shape}")
    print(f"  Max difference: {max_diff:.3e}")
    print(f"  Mean difference: {mean_diff:.3e}")
    print(f"  Std difference: {std_diff:.3e}")
    print(f"  Max diff location: {max_idx}")
    print(f"  Value at max diff - Python: {arr1[max_idx]:.15f}")
    print(f"  Value at max diff - Binary: {arr2[max_idx]:.15f}")

    # Check relative error for non-zero values
    mask = arr2 != 0
    if np.any(mask):
        rel_errors = np.abs((arr1[mask] - arr2[mask]) / arr2[mask])
        max_rel_error = np.max(rel_errors)
        print(f"  Max relative error: {max_rel_error:.3e}")

    within_tolerance = max_diff < tolerance
    print(
        f"  Within tolerance ({tolerance:.0e}): {'✓ YES' if within_tolerance else '✗ NO'}"
    )

    return within_tolerance


def test_st_accuracy(data_file: str = "ROS_4.ascii") -> bool:
    """
    Test Single Timeseries DDA accuracy against binary.

    Args:
        data_file: Path to data file

    Returns:
        True if test passes
    """
    print("\n" + "=" * 60)
    print("Testing ST (Single Timeseries) DDA Accuracy")
    print("=" * 60)

    if not os.path.exists(data_file):
        print(f"Error: Data file {data_file} not found")
        return False

    # Load data
    Y = np.loadtxt(data_file)
    print(f"Data shape: {Y.shape}")

    # DDA parameters
    TAU = [32, 9]
    dm = 4
    order = 3
    WL = 2000
    WS = 1000

    # Run external binary DDA
    print("\nRunning external binary DDA...")
    MODEL = np.array([1, 2, 3])  # Standard ST model

    # Create temporary file for external DDA
    temp_st_file = "temp_st_test"
    cmd, ST_external = run_dda_st_external(
        data_file,
        temp_st_file,
        MODEL,
        TAU,
        dm=dm,
        DDAorder=order,
        nr_delays=2,
        WL=WL,
        WS=WS,
    )

    print(f"External command: {cmd}")
    print(f"External ST shape: {ST_external.shape}")

    # Run Python implementation
    print("\nRunning Python implementation...")
    ST_python = compute_st_multiple(Y, TAU, dm, order, WL, WS, return_dict=False)

    # Reshape to match external format (Julia column-major)
    WN = ST_python.shape[0]
    ST_python_reshaped = ST_python.reshape(WN, -1, order="F")

    print(f"Python ST shape: {ST_python.shape}")
    print(f"Python ST reshaped: {ST_python_reshaped.shape}")

    # Compare results
    st_match = compare_arrays(ST_python_reshaped, ST_external, "ST", tolerance=1e-17)

    # Clean up temporary files
    for ext in ["_ST", "_CT", "_MT", "_DE"]:
        if os.path.exists(temp_st_file + ext):
            os.remove(temp_st_file + ext)

    return st_match


def test_ct_accuracy(data_file: str = "ROS_4.ascii") -> bool:
    """
    Test Cross Timeseries DDA accuracy against binary.

    Args:
        data_file: Path to data file

    Returns:
        True if test passes
    """
    print("\n" + "=" * 60)
    print("Testing CT (Cross Timeseries) DDA Accuracy")
    print("=" * 60)

    if not os.path.exists(data_file):
        print(f"Error: Data file {data_file} not found")
        return False

    # Load data
    Y = np.loadtxt(data_file)
    print(f"Data shape: {Y.shape}")

    # DDA parameters
    TAU = [32, 9]
    dm = 4
    order = 3
    WL = 2000
    WS = 1000

    # Run external binary DDA
    print("\nRunning external binary DDA...")
    MODEL = np.array([1, 2, 3])  # Standard CT model

    # Create temporary file for external DDA
    temp_ct_file = "temp_ct_test"
    cmd, CT_external = run_dda_ct_external(
        data_file,
        temp_ct_file,
        MODEL,
        TAU,
        dm=dm,
        DDAorder=order,
        nr_delays=2,
        WL=WL,
        WS=WS,
    )

    print(f"External command: {cmd}")
    print(f"External CT shape: {CT_external.shape}")

    # Run Python implementation
    print("\nRunning Python implementation...")
    CT_python, channel_pairs = compute_ct_multiple(Y, TAU, dm, order, WL, WS)

    # Reshape to match external format (Julia column-major)
    WN = CT_python.shape[0]
    CT_python_reshaped = CT_python.reshape(WN, -1, order="F")

    print(f"Python CT shape: {CT_python.shape}")
    print(f"Python CT reshaped: {CT_python_reshaped.shape}")
    print(f"Channel pairs: {channel_pairs}")

    # Compare results
    ct_match = compare_arrays(CT_python_reshaped, CT_external, "CT", tolerance=1e-17)

    # Clean up temporary files
    for ext in ["_ST", "_CT", "_MT", "_DE"]:
        if os.path.exists(temp_ct_file + ext):
            os.remove(temp_ct_file + ext)

    return ct_match


def test_dictionary_output():
    """Test the dictionary output format."""
    print("\n" + "=" * 60)
    print("Testing Dictionary Output Format")
    print("=" * 60)

    # Create test data
    t = np.linspace(0, 100, 5000)
    test_data = np.sin(0.1 * t) + 0.1 * np.random.randn(len(t))

    # Test single channel
    result = compute_st_single(
        test_data,
        TAU=[10, 20],
        dm=4,
        order=3,
        WL=1000,
        WS=500,
        return_dict=True,
        sampling_rate=100.0,
        units="mV",
    )

    print("\nSingle Channel Dictionary Output:")
    print(f"  Keys: {list(result.keys())}")
    print(f"  Coefficients shape: {result['coefficients'].shape}")
    print(f"  Errors shape: {result['errors'].shape}")
    print(f"  Metadata: {result['metadata']}")
    print(f"  Model equation: {result['model_description']['equation']}")

    # Test multi-channel
    multi_data = np.column_stack(
        [np.sin(0.1 * t), np.cos(0.1 * t), np.sin(0.2 * t)]
    ) + 0.1 * np.random.randn(len(t), 3)

    result_multi = compute_st_multiple(
        multi_data,
        TAU=[10, 20],
        dm=4,
        order=3,
        WL=1000,
        WS=500,
        return_dict=True,
        sampling_rate=250.0,
        units="μV",
        channel_names=["Ch1", "Ch2", "Ch3"],
    )

    print("\nMulti-Channel Dictionary Output:")
    print(f"  Coefficients shape: {result_multi['coefficients'].shape}")
    print(f"  Number of channels: {result_multi['metadata']['n_channels']}")
    print(f"  Channel names: {result_multi['metadata']['channel_names']}")

    # Verify backward compatibility
    array_result = compute_st_single(test_data, [10, 20], return_dict=False)
    print(f"\nBackward compatible array output shape: {array_result.shape}")

    return True


def main():
    """Main test function."""
    print("High-Precision DDA Accuracy Test Suite")
    print("Target tolerance: 1e-17")

    all_tests_pass = True

    # Test dictionary output format
    dict_test_pass = test_dictionary_output()
    all_tests_pass &= dict_test_pass

    # Check if we have binary executables
    if not (
        os.path.exists("run_DDA_AsciiEdf") or os.path.exists("run_DDA_AsciiEdf.exe")
    ):
        print("\n⚠️  Warning: DDA binary executable not found")
        print("Cannot perform accuracy comparison with external binary")
        print("Please ensure run_DDA_AsciiEdf is in the current directory")
        return False

    # Test ST accuracy
    st_test_pass = test_st_accuracy()
    all_tests_pass &= st_test_pass

    # Test CT accuracy
    ct_test_pass = test_ct_accuracy()
    all_tests_pass &= ct_test_pass

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Dictionary output test: {'✓ PASS' if dict_test_pass else '✗ FAIL'}")
    print(f"ST accuracy test: {'✓ PASS' if st_test_pass else '✗ FAIL'}")
    print(f"CT accuracy test: {'✓ PASS' if ct_test_pass else '✗ FAIL'}")
    print(f"\nOverall: {'✓ ALL TESTS PASS' if all_tests_pass else '✗ TESTS FAILED'}")

    return all_tests_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
