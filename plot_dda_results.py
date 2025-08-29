#!/usr/bin/env python3
"""
Plot DDA analysis results from Rust implementation.

This script reads DDA ST (single-timeseries) results and creates visualizations
showing the temporal evolution of the a1 coefficient.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path


def load_dda_data(filename):
    """
    Load DDA ST results from file.

    Expected format: window_idx channel_idx time a1 a2 a3 error
    """
    try:
        data = np.loadtxt(filename)
        print(f"Loaded data shape: {data.shape}")
        print("Columns: window_idx, channel_idx, a1, a2, a3, error")
        return data
    except Exception as e:
        print(f"Error loading data from {filename}: {e}")
        sys.exit(1)


def create_time_series_plot(data, output_file=None):
    """
    Create a time series plot showing a1 coefficient vs time for each channel on one plot.
    """
    # Extract unique channels
    channels = np.unique(data[:, 1])
    n_channels = len(channels)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Use a colormap with enough distinct colors
    if n_channels <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_channels))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, n_channels))

    for i, channel in enumerate(channels):
        # Filter data for this channel
        channel_mask = data[:, 1] == channel
        channel_data = data[channel_mask]

        # Sort by window index to ensure proper time ordering
        sort_idx = np.argsort(channel_data[:, 0])
        channel_data = channel_data[sort_idx]

        # Extract time (window index) and a1 coefficient
        time = channel_data[:, 0]  # Window index as time
        a1_coeff = channel_data[:, 2]  # a1 coefficient

        ax.plot(
            time, a1_coeff, "-", color=colors[i], linewidth=1.5, 
            alpha=0.8, label=f"Channel {int(channel)}"
        )

    ax.set_xlabel("Time Window Index", fontsize=12)
    ax.set_ylabel("a1 coefficient", fontsize=12)
    ax.set_title("DDA Single-Timeseries Analysis: a1 Coefficient Evolution", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    if n_channels <= 20:  # Only show legend if not too many channels
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Time series plot saved to: {output_file}")
    else:
        plt.show()


def create_heatmap(data, output_file=None):
    """
    Create a heatmap showing a1 coefficient (3rd column) with time on x-axis and channels on y-axis.
    """
    # Get unique channels and time windows
    channels = np.unique(data[:, 1])
    windows = np.unique(data[:, 0])

    # Create matrix for heatmap: channels (rows) x time_windows (columns)
    heatmap_data = np.full((len(channels), len(windows)), np.nan)

    for i, channel in enumerate(channels):
        channel_mask = data[:, 1] == channel
        channel_data = data[channel_mask]

        for row in channel_data:
            window_idx = int(row[0])
            a1_coeff = row[2]  # 3rd column (a1 coefficient)

            # Find position in matrix
            time_pos = np.where(windows == window_idx)[0]
            if len(time_pos) > 0:
                heatmap_data[i, time_pos[0]] = a1_coeff

    # Create heatmap with time on x-axis
    fig, ax = plt.subplots(figsize=(max(10, len(windows) * 0.3), max(6, len(channels) * 0.2)))

    im = ax.imshow(heatmap_data, aspect="auto", cmap="RdBu_r", interpolation="nearest", origin='lower')

    # Customize axes - time on x-axis
    ax.set_xlabel("Time Window Index", fontsize=12)
    ax.set_ylabel("Channel", fontsize=12)
    ax.set_title("DDA a1 Coefficient Heatmap (Time vs Channels)", fontsize=14, pad=20)

    # Set time window labels on x-axis
    n_time_ticks = min(20, len(windows))
    time_tick_indices = np.linspace(0, len(windows) - 1, n_time_ticks, dtype=int)
    ax.set_xticks(time_tick_indices)
    ax.set_xticklabels([f"{int(windows[i])}" for i in time_tick_indices], rotation=45)

    # Set channel labels on y-axis (limit if too many channels)
    if len(channels) <= 50:
        ax.set_yticks(range(len(channels)))
        ax.set_yticklabels([f"Ch{int(ch)}" for ch in channels], fontsize=8)
    else:
        # Show every 10th channel label for large datasets
        n_ch_ticks = min(20, len(channels))
        ch_tick_indices = np.linspace(0, len(channels) - 1, n_ch_ticks, dtype=int)
        ax.set_yticks(ch_tick_indices)
        ax.set_yticklabels([f"Ch{int(channels[i])}" for i in ch_tick_indices], fontsize=8)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("a1 coefficient", rotation=270, labelpad=20, fontsize=12)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Heatmap saved to: {output_file}")
    else:
        plt.show()


def print_data_summary(data):
    """Print summary statistics of the data."""
    channels = np.unique(data[:, 1])
    windows = np.unique(data[:, 0])

    print("\nData Summary:")
    print(f"  Number of channels: {len(channels)}")
    print(f"  Channel IDs: {channels}")
    print(f"  Number of time windows: {len(windows)}")
    print(f"  Time window range: {windows[0]} to {windows[-1]}")


def main():
    parser = argparse.ArgumentParser(description="Plot DDA analysis results")
    parser.add_argument("input_file", help="DDA ST results file")
    parser.add_argument(
        "--output-dir",
        "-o",
        default=".",
        help="Output directory for plots (default: current directory)",
    )
    parser.add_argument(
        "--no-heatmap", action="store_true", help="Skip heatmap generation"
    )
    parser.add_argument(
        "--no-timeseries", action="store_true", help="Skip time series plot generation"
    )
    parser.add_argument(
        "--show", action="store_true", help="Show plots interactively instead of saving"
    )

    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)

    # Load data
    print(f"Loading DDA results from: {args.input_file}")
    data = load_dda_data(args.input_file)

    # Print data summary
    print_data_summary(data)

    # Prepare output filenames
    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    base_name = input_path.stem

    # Create plots
    if not args.no_timeseries:
        if args.show:
            create_time_series_plot(data)
        else:
            ts_output = output_dir / f"{base_name}_timeseries.png"
            create_time_series_plot(data, ts_output)

    if not args.no_heatmap:
        if args.show:
            create_heatmap(data)
        else:
            hm_output = output_dir / f"{base_name}_heatmap.png"
            create_heatmap(data, hm_output)

    print("\nPlotting complete!")


if __name__ == "__main__":
    main()
