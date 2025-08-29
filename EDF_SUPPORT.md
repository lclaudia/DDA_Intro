# EDF File Support for DDA

The Rust DDA implementation now supports EDF (European Data Format) files through a Python conversion utility.

## Quick Start

### 1. Convert EDF to ASCII
```bash
python3 edf_to_ascii.py your_file.edf
```

### 2. Run DDA
```bash
./rs/target/release/dda st --input your_file.txt -o results.txt
```

### 3. Visualize Results
```bash
python3 plot_dda_results.py results.txt -o plots/
```

## Requirements

Install the required Python dependencies:
```bash
pip install pyedflib numpy matplotlib
```

## EDF Converter Options

```bash
# Basic conversion
python3 edf_to_ascii.py input.edf

# Custom output file
python3 edf_to_ascii.py input.edf -o custom_output.txt

# Limit duration (useful for large files)
python3 edf_to_ascii.py input.edf --max-duration 60

# Choose backend (pyedflib or mne)
python3 edf_to_ascii.py input.edf --backend pyedflib
```

## Example with Patient Data

The provided patient EEG file (`patient1_S05__01_03.edf`) contains:
- **156 channels** (LAT1-8, LPT1-8, LPF1-8, etc.)
- **500 Hz sampling rate**
- **240.4 seconds duration**
- **120,190 total samples**

### Convert and analyze:
```bash
# Convert full file
python3 edf_to_ascii.py patient1_S05__01_03.edf

# For demo purposes, use a subset
head -5000 patient1_S05__01_03.txt | cut -d' ' -f1-4 > demo_subset.txt

# Run DDA analysis
./rs/target/release/dda st --input demo_subset.txt -o patient_results.txt

# Create plots
python3 plot_dda_results.py patient_results.txt -o patient_plots/
```

## DDA Analysis Results

The analysis produces:
- **a1, a2, a3**: DDA coefficients representing dynamical properties
- **Error**: Residual fitting error
- **Time windows**: Temporal evolution of coefficients

### Visualization Output:
- **Time series plot**: Shows coefficient evolution over time
- **Heatmap**: Overview of coefficients across channels and time
- **Statistics**: Mean and standard deviation summaries

## File Formats

### Input EDF Properties:
- Multi-channel biomedical data
- Standard EDF/EDF+ format
- Any sampling rate (converted to uniform grid)

### Output ASCII Format:
- Space-separated values
- Rows: time samples
- Columns: channels
- Scientific notation (e.g., `1.234567e-02`)

## Error Handling

If you try to use EDF files directly with the Rust DDA tool:
```
Error: EDF files are not directly supported. Please convert to ASCII format first.
Use: python3 edf_to_ascii.py your_file.edf -o your_file.txt
```

## Tips

1. **Large files**: Use `--max-duration` to limit conversion time
2. **Memory usage**: Consider processing channel subsets for very large datasets  
3. **DDA parameters**: Adjust `--wl` (window length) and `--ws` (window shift) based on your data characteristics
4. **Visualization**: Use `--show` flag for interactive plots

## Troubleshooting

### Missing dependencies:
```bash
pip install pyedflib numpy matplotlib
```

### Alternative EDF reader:
```bash
pip install mne
python3 edf_to_ascii.py file.edf --backend mne
```

### Large file handling:
```bash
# Process first 60 seconds only
python3 edf_to_ascii.py large_file.edf --max-duration 60
```