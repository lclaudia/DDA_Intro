"""
Dynamical Ergodicity (DE) DDA Analysis Module

This module provides functions for computing dynamical ergodicity measures
from DDA analysis results. It quantifies the relationship between single
timeseries and cross-timeseries analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple
from numpy.typing import NDArray


def compute_dynamical_ergodicity(
    ST: NDArray,
    CT: NDArray,
    channel_pairs: NDArray,
) -> NDArray:
    """
    Compute dynamical ergodicity matrix from ST and CT results.
    
    The dynamical ergodicity measure quantifies how well the coupling
    between two timeseries can be explained by their individual structures.
    
    Args:
        ST: Single timeseries results, shape (WN, 4, n_channels)
            Last dimension (index 3) contains the error values
        CT: Cross-timeseries results, shape (WN, 4, n_pairs)
            Last dimension (index 3) contains the error values
        channel_pairs: Array of channel pairs, shape (n_pairs, 2)
            0-based indices indicating which channels are paired
    
    Returns:
        Ergodicity matrix E of shape (n_channels, n_channels)
        E[i,j] = |mean([st_i, st_j])/ct_ij - 1|
    """
    n_channels = ST.shape[2]
    
    # Compute mean errors over windows
    st = np.mean(ST[:, -1, :], axis=0)  # Mean over windows, last column (errors)
    ct = np.mean(CT[:, -1, :], axis=0)  # Mean over windows, last column (errors)
    
    # Initialize ergodicity matrix
    E = np.full((n_channels, n_channels), np.nan)
    
    # Fill the matrix with ergodicity values
    for n_pair, (ch1, ch2) in enumerate(channel_pairs):
        # Ergodicity measure: |mean([st_i, st_j])/ct_ij - 1|
        E[ch1, ch2] = abs(np.mean([st[ch1], st[ch2]]) / ct[n_pair] - 1)
        E[ch2, ch1] = E[ch1, ch2]  # Symmetric matrix
    
    return E


def plot_ergodicity_heatmap(
    E: NDArray,
    channel_labels: Optional[list] = None,
    title: str = "Dynamical Ergodicity Heatmap",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = "ergodicity_heatmap.png",
    cmap: str = "viridis",
    annot: bool = True,
    fmt: str = ".2e",
) -> None:
    """
    Create and display a heatmap of the dynamical ergodicity matrix.
    
    Args:
        E: Ergodicity matrix
        channel_labels: Optional labels for channels
        title: Plot title
        figsize: Figure size as (width, height)
        save_path: Path to save the figure (None to not save)
        cmap: Colormap name
        annot: Whether to annotate cells with values
        fmt: Format string for annotations
    """
    plt.figure(figsize=figsize)
    
    # Create heatmap
    # Handle None labels
    xticklabels = channel_labels if channel_labels is not None else True
    yticklabels = channel_labels if channel_labels is not None else True
    
    sns.heatmap(
        E,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        square=True,
        cbar_kws={"label": "Ergodicity"},
    )
    
    plt.title(title)
    plt.xlabel("Channel")
    plt.ylabel("Channel")
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def analyze_ergodicity_statistics(
    E: NDArray,
    threshold: float = 0.1,
) -> dict:
    """
    Compute statistics about the ergodicity matrix.
    
    Args:
        E: Ergodicity matrix
        threshold: Threshold for considering channels as ergodic
    
    Returns:
        Dictionary containing various statistics:
        - mean: Mean ergodicity value
        - std: Standard deviation
        - min: Minimum value (excluding diagonal)
        - max: Maximum value
        - n_ergodic: Number of ergodic pairs (below threshold)
        - ergodic_pairs: List of ergodic channel pairs
    """
    # Extract upper triangle (excluding diagonal)
    upper_triangle_indices = np.triu_indices_from(E, k=1)
    upper_values = E[upper_triangle_indices]
    
    # Remove NaN values
    valid_values = upper_values[~np.isnan(upper_values)]
    
    # Find ergodic pairs
    ergodic_mask = valid_values < threshold
    ergodic_pairs = []
    
    for idx, is_ergodic in enumerate(ergodic_mask):
        if is_ergodic:
            i, j = upper_triangle_indices[0][idx], upper_triangle_indices[1][idx]
            ergodic_pairs.append((i, j))
    
    stats = {
        "mean": np.mean(valid_values),
        "std": np.std(valid_values),
        "min": np.min(valid_values),
        "max": np.max(valid_values),
        "n_ergodic": len(ergodic_pairs),
        "ergodic_pairs": ergodic_pairs,
        "total_pairs": len(valid_values),
        "ergodic_fraction": len(ergodic_pairs) / len(valid_values) if len(valid_values) > 0 else 0,
    }
    
    return stats


def compare_with_external_de(
    E_computed: NDArray,
    E_external: NDArray,
    tolerance: float = 1e-15,
) -> Tuple[float, bool]:
    """
    Compare computed ergodicity matrix with external results.
    
    Args:
        E_computed: Computed ergodicity matrix
        E_external: External/reference ergodicity matrix
        tolerance: Tolerance for considering values equal
    
    Returns:
        Tuple of (mean_error, is_within_tolerance)
    """
    # Compute difference only for non-NaN values
    valid_mask = ~np.isnan(E_computed) & ~np.isnan(E_external)
    
    if not np.any(valid_mask):
        return np.nan, False
    
    diff = E_computed[valid_mask] - E_external[valid_mask]
    mean_error = np.mean(np.abs(diff))
    is_within_tolerance = np.all(np.abs(diff) < tolerance)
    
    return mean_error, is_within_tolerance


def run_full_de_analysis(
    Y: NDArray,
    TAU: list,
    dm: int = 4,
    order: int = 3,
    WL: int = 2000,
    WS: int = 1000,
    plot: bool = True,
    save_plot: Optional[str] = "ergodicity_heatmap.png",
) -> Tuple[NDArray, dict]:
    """
    Run complete dynamical ergodicity analysis on data.
    
    Args:
        Y: Input data matrix (samples x channels)
        TAU: List of two delay values [tau1, tau2]
        dm: Derivative method parameter (default: 4)
        order: Order of DDA (default: 3)
        WL: Window length (default: 2000)
        WS: Window shift (default: 1000)
        plot: Whether to create heatmap plot
        save_plot: Path to save plot
    
    Returns:
        Tuple of (ergodicity matrix, statistics dictionary)
    """
    from dda_st import compute_st_multiple
    from dda_ct import compute_ct_multiple
    
    # Compute ST for all channels
    ST = compute_st_multiple(Y, TAU, dm, order, WL, WS)
    
    # Compute CT for all channel pairs
    CT, channel_pairs = compute_ct_multiple(Y, TAU, dm, order, WL, WS)
    
    # Compute ergodicity matrix
    E = compute_dynamical_ergodicity(ST, CT, channel_pairs)
    
    # Compute statistics
    stats = analyze_ergodicity_statistics(E)
    
    # Create plot if requested
    if plot:
        plot_ergodicity_heatmap(E, save_path=save_plot)
    
    return E, stats