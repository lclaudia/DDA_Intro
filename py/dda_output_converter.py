"""
DDA Output Converter Module

This module provides standardized functions to convert DDA algorithm outputs
into a consistent dictionary format for interoperability across different
applications and domains.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from numpy.typing import NDArray
import pandas as pd


def convert_dda_output(
    coefficients_matrix: NDArray,
    algorithm: str,
    delays: Optional[List[float]] = None,
    window_length: Optional[int] = None,
    window_shift: Optional[int] = None,
    derivative_method: Optional[int] = None,
    order: Optional[int] = None,
    sampling_rate: Optional[float] = None,
    units: Optional[str] = None,
    channel_names: Optional[List[str]] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convert DDA output matrix to standardized dictionary format.

    This function handles different DDA algorithm outputs and converts them
    into a consistent structure suitable for various applications.

    Args:
        coefficients_matrix: Raw output from DDA algorithm
            - For ST: shape (n_windows, 4) or (n_windows, 4, n_channels)
            - For MT: shape (n_windows, n_coeffs) or (n_windows, n_coeffs, n_channels)
        algorithm: Name of the DDA algorithm (e.g., 'DDA_ST', 'DDA_MT', 'DDA_CT')
        delays: List of delay values used
        window_length: Length of analysis window
        window_shift: Shift between windows
        derivative_method: Method used for derivative calculation
        order: Order of the DDA analysis
        sampling_rate: Sampling rate of the data (Hz)
        units: Physical units of the data
        channel_names: Names of channels for multi-channel data
        additional_metadata: Any additional algorithm-specific metadata

    Returns:
        Standardized dictionary containing coefficients, errors, and metadata
    """
    # Determine data structure
    if coefficients_matrix.ndim == 2:
        # Single channel
        n_windows, n_features = coefficients_matrix.shape
        n_channels = 1
        multi_channel = False
    elif coefficients_matrix.ndim == 3:
        # Multi channel
        n_windows, n_features, n_channels = coefficients_matrix.shape
        multi_channel = True
    else:
        raise ValueError(f"Unexpected matrix dimensions: {coefficients_matrix.ndim}")

    # Extract coefficients and errors based on algorithm
    if algorithm == "DDA_ST":
        if multi_channel:
            coefficients = coefficients_matrix[:, :3, :]
            errors = coefficients_matrix[:, 3, :]
        else:
            coefficients = coefficients_matrix[:, :3]
            errors = coefficients_matrix[:, 3]

        model_description = {
            "equation": "dx/dt = c0*x(t-τ1) + c1*x(t-τ2) + c2*x(t-τ1)³",
            "coefficients_mapping": {
                "c0": "Linear coefficient for delay τ1",
                "c1": "Linear coefficient for delay τ2",
                "c2": "Cubic nonlinearity coefficient",
            },
        }

    elif algorithm == "DDA_MT":
        # For MT, determine how many coefficients vs error columns
        n_coeffs = n_features - n_channels if multi_channel else n_features - 1

        if multi_channel:
            coefficients = coefficients_matrix[:, :n_coeffs, :]
            errors = coefficients_matrix[:, n_coeffs:, :]
        else:
            coefficients = coefficients_matrix[:, :n_coeffs]
            errors = coefficients_matrix[:, n_coeffs]

        model_description = {
            "equation": "Multi-delay differential equation with cross-terms",
            "coefficients_mapping": _generate_mt_coefficient_mapping(n_coeffs, delays),
        }

    else:
        # Generic handling for other algorithms
        # Assume last column/channel is error
        if multi_channel:
            coefficients = coefficients_matrix[:, :-1, :]
            errors = coefficients_matrix[:, -1, :]
        else:
            coefficients = coefficients_matrix[:, :-1]
            errors = coefficients_matrix[:, -1]

        model_description = {
            "equation": f"{algorithm} differential equation",
            "coefficients_mapping": {
                f"c{i}": f"Coefficient {i}" for i in range(coefficients.shape[1])
            },
        }

    # Build metadata
    metadata = {
        "method": algorithm,
        "n_windows": n_windows,
        "n_channels": n_channels,
        "multi_channel": multi_channel,
    }

    # Add optional metadata if provided
    optional_fields = {
        "delays": delays,
        "window_length": window_length,
        "window_shift": window_shift,
        "derivative_method": derivative_method,
        "order": order,
        "sampling_rate": sampling_rate,
        "units": units,
        "channel_names": channel_names,
    }

    for key, value in optional_fields.items():
        if value is not None:
            metadata[key] = value

    # Add any additional metadata
    if additional_metadata:
        metadata.update(additional_metadata)

    # Build window information if possible
    window_info = {}
    if window_shift is not None:
        window_info["start_indices"] = np.arange(n_windows) * window_shift
        if window_length is not None:
            window_info["end_indices"] = window_info["start_indices"] + window_length

    # Construct output dictionary
    output = {
        "coefficients": coefficients,
        "errors": errors,
        "metadata": metadata,
        "model_description": model_description,
    }

    if window_info:
        output["window_info"] = window_info

    return output


def _generate_mt_coefficient_mapping(
    n_coeffs: int, delays: Optional[List[float]]
) -> Dict[str, str]:
    """Generate coefficient mapping for MT algorithm."""
    mapping = {}

    if delays and len(delays) >= 2:
        # Known structure with delays
        base_terms = len(delays)
        mapping.update(
            {f"c{i}": f"Linear term for delay τ{i+1}" for i in range(base_terms)}
        )

        # Add cross-terms and nonlinear terms
        idx = base_terms
        for i in range(base_terms):
            for j in range(i, base_terms):
                if idx < n_coeffs:
                    mapping[f"c{idx}"] = f"Cross-term τ{i+1}×τ{j+1}"
                    idx += 1
    else:
        # Generic mapping
        mapping = {f"c{i}": f"Coefficient {i}" for i in range(n_coeffs)}

    return mapping


def convert_to_dataframe(
    dda_output: Dict[str, Any], time_column: bool = True, long_format: bool = False
) -> "pd.DataFrame":
    """
    Convert standardized DDA output to pandas DataFrame.

    Args:
        dda_output: Output from convert_dda_output()
        time_column: Add time/window index column
        long_format: Use long format (good for plotting) vs wide format

    Returns:
        DataFrame with DDA results
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for DataFrame conversion")

    coefficients = dda_output["coefficients"]
    errors = dda_output["errors"]
    metadata = dda_output["metadata"]

    n_windows = metadata["n_windows"]
    n_channels = metadata.get("n_channels", 1)
    multi_channel = metadata.get("multi_channel", False)

    if long_format:
        # Create long format DataFrame
        data_list = []

        for window_idx in range(n_windows):
            for channel_idx in range(n_channels):
                row = {"window": window_idx}

                if time_column and "window_info" in dda_output:
                    row["start_index"] = dda_output["window_info"]["start_indices"][
                        window_idx
                    ]

                if multi_channel:
                    row["channel"] = channel_idx
                    if "channel_names" in metadata:
                        row["channel_name"] = metadata["channel_names"][channel_idx]

                    # Add coefficients
                    for coeff_idx in range(coefficients.shape[1]):
                        row[f"c{coeff_idx}"] = coefficients[
                            window_idx, coeff_idx, channel_idx
                        ]
                    row["error"] = errors[window_idx, channel_idx]
                else:
                    # Add coefficients
                    for coeff_idx in range(coefficients.shape[1]):
                        row[f"c{coeff_idx}"] = coefficients[window_idx, coeff_idx]
                    row["error"] = errors[window_idx]

                data_list.append(row)

        df = pd.DataFrame(data_list)

    else:
        # Create wide format DataFrame
        if multi_channel:
            # Multi-channel wide format
            columns = []
            data = []

            for channel_idx in range(n_channels):
                channel_name = (
                    metadata.get(
                        "channel_names", [f"ch{i}" for i in range(n_channels)]
                    )[channel_idx]
                    if "channel_names" in metadata
                    else f"ch{channel_idx}"
                )

                # Add coefficient columns
                for coeff_idx in range(coefficients.shape[1]):
                    columns.append(f"{channel_name}_c{coeff_idx}")
                    data.append(coefficients[:, coeff_idx, channel_idx])

                # Add error column
                columns.append(f"{channel_name}_error")
                data.append(errors[:, channel_idx])

            df = pd.DataFrame(np.column_stack(data), columns=columns)

        else:
            # Single channel wide format
            data_dict = {}

            # Add coefficients
            for coeff_idx in range(coefficients.shape[1]):
                data_dict[f"c{coeff_idx}"] = coefficients[:, coeff_idx]

            # Add error
            data_dict["error"] = errors

            df = pd.DataFrame(data_dict)

        # Add window index
        if time_column:
            df.insert(0, "window", np.arange(n_windows))
            if "window_info" in dda_output:
                df.insert(1, "start_index", dda_output["window_info"]["start_indices"])

    return df


def export_to_file(
    dda_output: Dict[str, Any],
    filename: str,
    format: str = "json",
    include_metadata: bool = True,
) -> None:
    """
    Export standardized DDA output to various file formats.

    Args:
        dda_output: Output from convert_dda_output()
        filename: Output filename (without extension)
        format: Output format ('json', 'hdf5', 'mat', 'csv', 'npz')
        include_metadata: Include metadata in export
    """
    import os

    if format == "json":
        import json

        # Convert numpy arrays to lists for JSON serialization
        json_output = {}
        for key, value in dda_output.items():
            if isinstance(value, np.ndarray):
                json_output[key] = value.tolist()
            elif isinstance(value, dict):
                json_output[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        json_output[key][k] = v.tolist()
                    else:
                        json_output[key][k] = v
            else:
                json_output[key] = value

        with open(f"{filename}.json", "w") as f:
            json.dump(json_output, f, indent=2)

    elif format == "hdf5":
        try:
            import h5py

            with h5py.File(f"{filename}.h5", "w") as f:
                # Save arrays
                f.create_dataset("coefficients", data=dda_output["coefficients"])
                f.create_dataset("errors", data=dda_output["errors"])

                # Save metadata as attributes
                if include_metadata:
                    for key, value in dda_output["metadata"].items():
                        if value is not None:
                            f.attrs[key] = value
        except ImportError:
            print(f"Warning: h5py not installed. Saving as NPZ instead.")
            export_to_file(
                dda_output, filename, format="npz", include_metadata=include_metadata
            )

    elif format == "mat":
        from scipy.io import savemat

        mat_dict = {
            "coefficients": dda_output["coefficients"],
            "errors": dda_output["errors"],
        }

        if include_metadata:
            mat_dict["metadata"] = dda_output["metadata"]

        savemat(f"{filename}.mat", mat_dict)

    elif format == "csv":
        df = convert_to_dataframe(dda_output)
        df.to_csv(f"{filename}.csv", index=False)

        # Save metadata separately
        if include_metadata:
            import json

            with open(f"{filename}_metadata.json", "w") as f:
                json.dump(dda_output["metadata"], f, indent=2)

    elif format == "npz":
        save_dict = {
            "coefficients": dda_output["coefficients"],
            "errors": dda_output["errors"],
        }

        if include_metadata:
            # Convert metadata to structured array for npz
            save_dict["metadata"] = np.array([str(dda_output["metadata"])])

        np.savez_compressed(f"{filename}.npz", **save_dict)

    else:
        raise ValueError(f"Unsupported format: {format}")
