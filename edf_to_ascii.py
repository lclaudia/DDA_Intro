#!/usr/bin/env python3
"""
Convert EDF files to ASCII format for use with DDA analysis.

This script reads EDF files and converts them to space-separated ASCII format
that can be read by the Rust DDA implementation.
"""

import numpy as np
import argparse
import sys
from pathlib import Path

try:
    import mne

    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

try:
    import pyedflib

    PYEDFLIB_AVAILABLE = True
except ImportError:
    PYEDFLIB_AVAILABLE = False


def convert_edf_with_pyedflib(edf_file, output_file, max_duration=None):
    """Convert EDF file to ASCII using pyedflib."""
    print(f"Reading EDF file: {edf_file}")

    with pyedflib.EdfReader(str(edf_file)) as f:
        n_channels = f.signals_in_file
        channel_names = f.getSignalLabels()
        sample_rates = [f.getSampleFrequency(i) for i in range(n_channels)]

        print("EDF info:")
        print(f"  Channels: {n_channels}")
        print(f"  Channel names: {channel_names}")
        print(f"  Sample rates: {sample_rates} Hz")
        print(f"  Duration: {f.file_duration:.1f} seconds")

        # Check if all channels have the same sample rate
        if len(set(sample_rates)) > 1:
            print(
                "Warning: Channels have different sample rates. Using the first channel's rate."
            )

        sample_rate = sample_rates[0]

        # Read all signals
        signals = []
        for i in range(n_channels):
            signal = f.readSignal(i)
            if max_duration:
                max_samples = int(max_duration * sample_rate)
                signal = signal[:max_samples]
            signals.append(signal)

        # Find the minimum length (in case channels have different lengths)
        min_length = min(len(signal) for signal in signals)
        signals = [signal[:min_length] for signal in signals]

        # Convert to array format (samples x channels)
        data = np.column_stack(signals)

    print(f"Data shape: {data.shape} (samples x channels)")
    print(f"Saving to: {output_file}")

    # Save as space-separated values
    np.savetxt(output_file, data, fmt="%.6e", delimiter=" ")
    print("Conversion complete!")

    return data.shape


def convert_edf_with_mne(edf_file, output_file, max_duration=None):
    """Convert EDF file to ASCII using MNE."""
    print(f"Reading EDF file: {edf_file}")

    raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)

    print("EDF info:")
    print(f"  Channels: {len(raw.ch_names)}")
    print(f"  Channel names: {raw.ch_names}")
    print(f"  Sample rate: {raw.info['sfreq']} Hz")
    print(f"  Duration: {raw.times[-1]:.1f} seconds")

    # Get data
    data = raw.get_data().T  # Transpose to get (samples x channels)

    if max_duration:
        max_samples = int(max_duration * raw.info["sfreq"])
        data = data[:max_samples, :]

    print(f"Data shape: {data.shape} (samples x channels)")
    print(f"Saving to: {output_file}")

    # Save as space-separated values
    np.savetxt(output_file, data, fmt="%.6e", delimiter=" ")
    print("Conversion complete!")

    return data.shape


def main():
    parser = argparse.ArgumentParser(
        description="Convert EDF files to ASCII format for DDA analysis"
    )
    parser.add_argument("input_file", help="Input EDF file")
    parser.add_argument(
        "-o", "--output", help="Output ASCII file (default: input_file.txt)"
    )
    parser.add_argument(
        "--max-duration", type=float, help="Maximum duration in seconds to convert"
    )
    parser.add_argument(
        "--backend",
        choices=["pyedflib", "mne", "auto"],
        default="auto",
        help="Backend to use for reading EDF files",
    )

    args = parser.parse_args()

    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)

    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = input_file.with_suffix(".txt")

    # Check available backends
    if args.backend == "auto":
        if PYEDFLIB_AVAILABLE:
            backend = "pyedflib"
        elif MNE_AVAILABLE:
            backend = "mne"
        else:
            print("Error: No EDF reading backend available.")
            print("Please install one of the following:")
            print("  pip install pyedflib")
            print("  pip install mne")
            sys.exit(1)
    else:
        backend = args.backend
        if backend == "pyedflib" and not PYEDFLIB_AVAILABLE:
            print("Error: pyedflib not available. Install with: pip install pyedflib")
            sys.exit(1)
        elif backend == "mne" and not MNE_AVAILABLE:
            print("Error: mne not available. Install with: pip install mne")
            sys.exit(1)

    print(f"Using backend: {backend}")

    try:
        if backend == "pyedflib":
            shape = convert_edf_with_pyedflib(
                input_file, output_file, args.max_duration
            )
        else:  # mne
            shape = convert_edf_with_mne(input_file, output_file, args.max_duration)

        print(f"\n✅ Success! Converted {shape[0]} samples x {shape[1]} channels")
        print(f"📄 Output file: {output_file}")
        print("\n💡 Now you can run DDA analysis with:")
        print(f"   ./rs/target/release/dda st --input {output_file} -o results_st.txt")

    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        print("\nTroubleshooting:")
        print("- Make sure the EDF file is valid")
        print("- Try a different backend with --backend option")
        print("- Check if you have the required dependencies installed")
        sys.exit(1)


if __name__ == "__main__":
    main()
