# DDA Rust Implementation

A portable Rust implementation of Delay Differential Analysis (DDA) algorithms.

## Installation

### Prerequisites

- Rust 1.70+ (install from https://rustup.rs/)
- BLAS library (OpenBLAS will be statically linked)

### Build

```bash
# Development build
cargo build

# Release build (optimized)
cargo build --release
```

The release binary will be in `target/release/dda`

## Usage

The `dda` tool provides several subcommands:

### Generate Synthetic Data

Generate coupled Roessler systems:

```bash
./target/release/dda generate -n 4 -o roessler_data.txt
```

Options:
- `-n`: Number of coupled systems (default: 4)
- `-o`: Output file path
- `--dt`: Integration time step (default: 0.05)
- `--length`: Number of integration steps (default: 20000)
- `--transient`: Transient steps to skip (default: 20000)
- `--delta`: Output every delta-th point (default: 2)
- `--seed`: Random seed for reproducibility

### Single Timeseries Analysis (ST)

```bash
./target/release/dda st -i data.txt --tau1 10 --tau2 30 -o st_results.txt
```

Options:
- `-i`: Input data file
- `--tau1`: First delay value (default: 10)
- `--tau2`: Second delay value (default: 30)
- `--dm`: Derivative method parameter (default: 4)
- `--order`: DDA order (default: 3)
- `--wl`: Window length (default: 2000)
- `--ws`: Window shift (default: 1000)
- `-o`: Output file path (optional)

### Cross Timeseries Analysis (CT)

```bash
./target/release/dda ct -i data.txt --tau1 10 --tau2 30 -o ct_results.txt
```

Options same as ST analysis. Requires at least 2 channels in input data.

### Dynamical Ergodicity Analysis (DE)

```bash
./target/release/dda de -i data.txt --tau1 10 --tau2 30 -o ergodicity.txt
```

Computes the full ergodicity matrix and statistics.

### Add Noise

```bash
./target/release/dda noise -i clean_data.txt --snr-db 20 -o noisy_data.txt
```

Options:
- `-i`: Input data file
- `--snr-db`: Signal-to-noise ratio in dB (default: 20.0)
- `-o`: Output file path
- `--column`: Specific column to add noise to (0-based index, default: all)

### Generate Test Data

Generate data matching the Julia implementation for testing:

```bash
./target/release/dda test -o ROS_4.ascii
```

## Cross-Platform Distribution

### Building for Different Platforms

From a Linux or macOS system with Rust installed:

```bash
# For Linux x86_64
cargo build --release --target x86_64-unknown-linux-gnu

# For macOS x86_64
cargo build --release --target x86_64-apple-darwin

# For macOS ARM64 (M1/M2)
cargo build --release --target aarch64-apple-darwin

# For Windows
cargo build --release --target x86_64-pc-windows-gnu
```

You may need to install the target first:
```bash
rustup target add x86_64-unknown-linux-gnu
```

### Creating Static Binaries

The Cargo.toml is configured to use static OpenBLAS linking, so the resulting binaries should be portable without requiring BLAS libraries on the target system.

## Data Format

Input data files should be ASCII text files with:
- Space or tab-separated values
- Each row is a time point
- Each column is a channel/variable

Example:
```
1.234 5.678 9.012
2.345 6.789 0.123
...
```

## Advantages Over Python/C

1. **Single binary**: No runtime dependencies
2. **Cross-platform**: Compile once for each target platform
3. **Performance**: Comparable to C, faster than Python
4. **Memory safe**: Rust's ownership system prevents memory errors
5. **Easy distribution**: Just copy the binary

## Testing

Compare with Python implementation:

```bash
# Generate test data
./target/release/dda test -o test_data.txt

# Run ST analysis
./target/release/dda st -i test_data.txt --tau1 10 --tau2 30 -o rust_st.txt

# Compare with Python results
python ../py/run_first_DDA.py
```

## License

[Same as original DDA implementation]