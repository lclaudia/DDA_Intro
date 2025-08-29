"""
Test script demonstrating the DDA output converter functionality.
"""

import numpy as np
from dda_st import compute_st_single, compute_st_multiple
from dda_output_converter import convert_to_dataframe, export_to_file


def test_single_channel():
    """Test single channel DDA output conversion."""
    print("Testing Single Channel DDA Output Conversion")
    print("=" * 50)
    
    # Generate synthetic data
    np.random.seed(42)
    data = np.cumsum(np.random.randn(10000))
    
    # Compute DDA with dictionary output
    result = compute_st_single(
        data, 
        TAU=[10, 20], 
        WL=1000, 
        WS=500,
        sampling_rate=100.0,  # 100 Hz
        units='mV'
    )
    
    print("\nDictionary Keys:", list(result.keys()))
    print(f"Coefficients shape: {result['coefficients'].shape}")
    print(f"Errors shape: {result['errors'].shape}")
    print(f"Number of windows: {result['metadata']['n_windows']}")
    print(f"Metadata: {result['metadata']}")
    print(f"\nModel description: {result['model_description']['equation']}")
    
    # Convert to DataFrame
    try:
        df = convert_to_dataframe(result)
        print(f"\nDataFrame shape: {df.shape}")
        print("DataFrame columns:", list(df.columns))
        print("\nFirst few rows:")
        print(df.head())
    except ImportError:
        print("\nNote: pandas not installed, skipping DataFrame conversion")
    
    # Export to different formats
    export_to_file(result, 'single_channel_dda', format='json')
    print("\nExported to single_channel_dda.json")
    
    return result


def test_multi_channel():
    """Test multi-channel DDA output conversion."""
    print("\n\nTesting Multi-Channel DDA Output Conversion")
    print("=" * 50)
    
    # Generate synthetic multi-channel data
    np.random.seed(42)
    n_channels = 3
    Y = np.cumsum(np.random.randn(10000, n_channels), axis=0)
    
    # Compute DDA with dictionary output
    result = compute_st_multiple(
        Y, 
        TAU=[10, 20], 
        WL=1000, 
        WS=500,
        sampling_rate=250.0,  # 250 Hz
        units='μV',
        channel_names=['Fz', 'Cz', 'Pz']  # EEG channel names
    )
    
    print("\nDictionary Keys:", list(result.keys()))
    print(f"Coefficients shape: {result['coefficients'].shape}")
    print(f"Errors shape: {result['errors'].shape}")
    print(f"Number of channels: {result['metadata']['n_channels']}")
    print(f"Channel names: {result['metadata']['channel_names']}")
    
    try:
        # Convert to DataFrame (wide format)
        df_wide = convert_to_dataframe(result, long_format=False)
        print(f"\nDataFrame (wide) shape: {df_wide.shape}")
        print("DataFrame columns:", list(df_wide.columns)[:10], "...")
        
        # Convert to DataFrame (long format)
        df_long = convert_to_dataframe(result, long_format=True)
        print(f"\nDataFrame (long) shape: {df_long.shape}")
        print("DataFrame columns:", list(df_long.columns))
        print("\nFirst few rows:")
        print(df_long.head())
    except ImportError:
        print("\nNote: pandas not installed, skipping DataFrame conversion")
    
    # Export to HDF5
    export_to_file(result, 'multi_channel_dda', format='hdf5')
    print("\nExported to multi_channel_dda.h5")
    
    return result


def test_backward_compatibility():
    """Test backward compatibility with array output."""
    print("\n\nTesting Backward Compatibility")
    print("=" * 50)
    
    # Generate test data
    np.random.seed(42)
    data = np.cumsum(np.random.randn(5000))
    
    # Get array output (backward compatible)
    array_result = compute_st_single(
        data, 
        TAU=[10, 20], 
        WL=1000, 
        WS=500,
        return_dict=False  # Returns numpy array
    )
    
    print(f"Array output shape: {array_result.shape}")
    print(f"Array output type: {type(array_result)}")
    print("\nFirst few rows of array:")
    print(array_result[:3])
    
    # Can still convert array manually if needed
    from dda_output_converter import convert_dda_output
    dict_result = convert_dda_output(
        array_result,
        algorithm='DDA_ST',
        delays=[10, 20],
        window_length=1000,
        window_shift=500
    )
    print("\nManually converted to dictionary format")
    print(f"Dictionary keys: {list(dict_result.keys())}")


if __name__ == "__main__":
    # Run all tests
    single_result = test_single_channel()
    multi_result = test_multi_channel()
    test_backward_compatibility()
    
    print("\n\nAll tests completed successfully!")