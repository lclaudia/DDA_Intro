use anyhow::Result;
use clap::{Parser, Subcommand};
use ndarray::{s, Array2};
use std::fs;
use std::path::PathBuf;

use dda::{
    compute_st_single, compute_st_multiple,
    compute_ct_multiple,
    run_full_de_analysis,
    generate_roessler_data, generate_test_data_matching_julia,
    add_noise,
};

#[derive(Parser)]
#[command(name = "dda")]
#[command(about = "Delay Differential Analysis tool", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate synthetic data using coupled Roessler systems
    Generate {
        /// Number of coupled systems
        #[arg(short = 'n', long, default_value = "4")]
        n_systems: usize,
        
        /// Output file path
        #[arg(short = 'o', long)]
        output: String,
        
        /// Integration time step
        #[arg(long, default_value = "0.05")]
        dt: f64,
        
        /// Number of integration steps
        #[arg(long, default_value = "20000")]
        length: usize,
        
        /// Transient steps to skip
        #[arg(long, default_value = "20000")]
        transient: usize,
        
        /// Output every delta-th point
        #[arg(long, default_value = "2")]
        delta: usize,
        
        /// Random seed
        #[arg(long)]
        seed: Option<u64>,
    },
    
    /// Run single-timeseries DDA analysis
    St {
        /// Input data file
        #[arg(short = 'i', long)]
        input: String,
        
        /// First delay value
        #[arg(long, default_value = "10")]
        tau1: usize,
        
        /// Second delay value
        #[arg(long, default_value = "30")]
        tau2: usize,
        
        /// Derivative method parameter
        #[arg(long, default_value = "4")]
        dm: usize,
        
        /// DDA order
        #[arg(long, default_value = "3")]
        order: usize,
        
        /// Window length
        #[arg(long, default_value = "2000")]
        wl: usize,
        
        /// Window shift
        #[arg(long, default_value = "1000")]
        ws: usize,
        
        /// Output file path
        #[arg(short = 'o', long)]
        output: Option<String>,
    },
    
    /// Run cross-timeseries DDA analysis
    Ct {
        /// Input data file
        #[arg(short = 'i', long)]
        input: String,
        
        /// First delay value
        #[arg(long, default_value = "10")]
        tau1: usize,
        
        /// Second delay value
        #[arg(long, default_value = "30")]
        tau2: usize,
        
        /// Derivative method parameter
        #[arg(long, default_value = "4")]
        dm: usize,
        
        /// DDA order
        #[arg(long, default_value = "3")]
        order: usize,
        
        /// Window length
        #[arg(long, default_value = "2000")]
        wl: usize,
        
        /// Window shift
        #[arg(long, default_value = "1000")]
        ws: usize,
        
        /// Output file path
        #[arg(short = 'o', long)]
        output: Option<String>,
    },
    
    /// Run full dynamical ergodicity analysis
    De {
        /// Input data file
        #[arg(short = 'i', long)]
        input: String,
        
        /// First delay value
        #[arg(long, default_value = "10")]
        tau1: usize,
        
        /// Second delay value
        #[arg(long, default_value = "30")]
        tau2: usize,
        
        /// Derivative method parameter
        #[arg(long, default_value = "4")]
        dm: usize,
        
        /// DDA order
        #[arg(long, default_value = "3")]
        order: usize,
        
        /// Window length
        #[arg(long, default_value = "2000")]
        wl: usize,
        
        /// Window shift
        #[arg(long, default_value = "1000")]
        ws: usize,
        
        /// Output file path
        #[arg(short = 'o', long)]
        output: Option<String>,
    },
    
    /// Add noise to a signal
    Noise {
        /// Input data file
        #[arg(short = 'i', long)]
        input: String,
        
        /// Signal-to-noise ratio in dB
        #[arg(long, default_value = "20.0")]
        snr_db: f64,
        
        /// Output file path
        #[arg(short = 'o', long)]
        output: String,
        
        /// Column index (0-based) to add noise to (default: all columns)
        #[arg(long)]
        column: Option<usize>,
    },
    
    /// Generate test data matching Julia implementation
    Test {
        /// Output file path
        #[arg(short = 'o', long, default_value = "ROS_4.ascii")]
        output: String,
    },
    
    /// Legacy DDA executable compatibility mode (matches run_DDA_AsciiEdf interface)
    Legacy {
        /// ASCII format flag
        #[arg(long = "ASCII", action = clap::ArgAction::SetTrue)]
        ascii: bool,
        
        /// Model specification
        #[arg(long = "MODEL", num_args = 1.., value_delimiter = ' ')]
        model: Vec<i32>,
        
        /// Delay values
        #[arg(long = "TAU", num_args = 1.., value_delimiter = ' ')]
        tau: Vec<usize>,
        
        /// Derivative method parameter
        #[arg(long)]
        dm: usize,
        
        /// DDA order
        #[arg(long)]
        order: usize,
        
        /// Number of delays
        #[arg(long)]
        nr_tau: usize,
        
        /// Input data filename
        #[arg(long = "DATA_FN")]
        data_fn: String,
        
        /// Output filename prefix
        #[arg(long = "OUT_FN")]
        out_fn: String,
        
        /// Window length
        #[arg(long = "WL")]
        wl: usize,
        
        /// Window shift
        #[arg(long = "WS")]
        ws: usize,
        
        /// Analysis selection [ST, CT, DE, reserved]
        #[arg(long = "SELECT", num_args = 4, value_delimiter = ' ')]
        select: Vec<u8>,
        
        /// Channel list for cross-timeseries analysis
        #[arg(long = "CH_list", num_args = 0.., value_delimiter = ' ')]
        ch_list: Vec<usize>,
        
        /// Window length for CT analysis
        #[arg(long = "WL_CT")]
        wl_ct: Option<usize>,
        
        /// Window shift for CT analysis
        #[arg(long = "WS_CT")]
        ws_ct: Option<usize>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            n_systems,
            output,
            dt,
            length,
            transient,
            delta,
            seed,
        } => {
            println!("Generating coupled Roessler systems data...");
            let (data, _params) = generate_roessler_data(
                n_systems,
                None,
                None,
                5.7,
                dt,
                length,
                transient,
                delta,
                Some(&output),
                seed,
            )?;
            println!("Generated data shape: ({}, {})", data.nrows(), data.ncols());
            println!("Data saved to: {}", output);
        }
        
        Commands::St {
            input,
            tau1,
            tau2,
            dm,
            order,
            wl,
            ws,
            output,
        } => {
            println!("Running single-timeseries DDA analysis...");
            let data = load_data(&input)?;
            let tau = vec![tau1, tau2];
            
            if data.ncols() == 1 {
                // Single channel
                let data_1d = data.column(0).to_owned();
                let st = compute_st_single(&data_1d, &tau, dm, order, wl, ws);
                
                if let Some(out_path) = output {
                    save_array_2d(&st, &out_path)?;
                    println!("ST results saved to: {}", out_path);
                } else {
                    println!("ST results shape: ({}, {})", st.nrows(), st.ncols());
                    println!("Mean error: {}", st.column(3).mean().unwrap());
                }
            } else {
                // Multiple channels
                let st = compute_st_multiple(&data, &tau, dm, order, wl, ws);
                
                if let Some(out_path) = output {
                    // Save as 2D array (flatten the 3D array)
                    let mut result = Vec::new();
                    for window in 0..st.dim().0 {
                        for channel in 0..st.dim().2 {
                            let mut row = vec![window as f64, channel as f64];
                            for feat in 0..st.dim().1 {
                                row.push(st[[window, feat, channel]]);
                            }
                            result.push(row);
                        }
                    }
                    let flat_st = Array2::from_shape_vec(
                        (result.len(), result[0].len()),
                        result.into_iter().flatten().collect()
                    )?;
                    save_array_2d(&flat_st, &out_path)?;
                    println!("ST results saved to: {}", out_path);
                } else {
                    println!("ST results shape: ({}, {}, {})", 
                             st.dim().0, st.dim().1, st.dim().2);
                    let mean_errors: Vec<f64> = (0..st.dim().2)
                        .map(|ch| st.slice(s![.., 3, ch]).mean().unwrap())
                        .collect();
                    println!("Mean errors per channel: {:?}", mean_errors);
                }
            }
        }
        
        Commands::Ct {
            input,
            tau1,
            tau2,
            dm,
            order,
            wl,
            ws,
            output,
        } => {
            println!("Running cross-timeseries DDA analysis...");
            let data = load_data(&input)?;
            let tau = vec![tau1, tau2];
            
            if data.ncols() < 2 {
                return Err(anyhow::anyhow!("CT analysis requires at least 2 channels"));
            }
            
            let (ct, pairs) = compute_ct_multiple(&data, &tau, dm, order, wl, ws, None);
            
            if let Some(out_path) = output {
                // Save as 2D array (flatten the 3D array)
                let mut result = Vec::new();
                for window in 0..ct.dim().0 {
                    for pair in 0..ct.dim().2 {
                        let mut row = vec![window as f64, pairs[[pair, 0]] as f64, pairs[[pair, 1]] as f64];
                        for feat in 0..ct.dim().1 {
                            row.push(ct[[window, feat, pair]]);
                        }
                        result.push(row);
                    }
                }
                let flat_ct = Array2::from_shape_vec(
                    (result.len(), result[0].len()),
                    result.into_iter().flatten().collect()
                )?;
                save_array_2d(&flat_ct, &out_path)?;
                println!("CT results saved to: {}", out_path);
            } else {
                println!("CT results shape: ({}, {}, {})", 
                         ct.dim().0, ct.dim().1, ct.dim().2);
                println!("Number of channel pairs: {}", pairs.nrows());
                let mean_errors: Vec<f64> = (0..ct.dim().2)
                    .map(|pair| ct.slice(s![.., 3, pair]).mean().unwrap())
                    .collect();
                println!("Mean errors per pair: {:?}", mean_errors);
            }
        }
        
        Commands::De {
            input,
            tau1,
            tau2,
            dm,
            order,
            wl,
            ws,
            output,
        } => {
            println!("Running dynamical ergodicity analysis...");
            let data = load_data(&input)?;
            let tau = vec![tau1, tau2];
            
            let (e, stats) = run_full_de_analysis(&data, &tau, dm, order, wl, ws)?;
            
            if let Some(out_path) = output {
                save_array_2d(&e, &out_path)?;
                println!("Ergodicity matrix saved to: {}", out_path);
            }
            
            println!("Ergodicity statistics:");
            println!("  Mean: {:.6}", stats.mean);
            println!("  Std: {:.6}", stats.std);
            println!("  Min: {:.6}", stats.min);
            println!("  Max: {:.6}", stats.max);
            println!("  Ergodic pairs: {}/{}", stats.n_ergodic, stats.total_pairs);
            println!("  Ergodic fraction: {:.3}", stats.ergodic_fraction);
        }
        
        Commands::Noise {
            input,
            snr_db,
            output,
            column,
        } => {
            println!("Adding noise to signal...");
            let data = load_data(&input)?;
            
            let noisy_data = if let Some(col_idx) = column {
                if col_idx >= data.ncols() {
                    return Err(anyhow::anyhow!("Column index {} out of range", col_idx));
                }
                let mut result = data.clone();
                let col = data.column(col_idx).to_owned();
                let noisy_col = add_noise(&col, snr_db);
                for (i, val) in noisy_col.iter().enumerate() {
                    result[[i, col_idx]] = *val;
                }
                result
            } else {
                // Add noise to all columns
                let mut result = data.clone();
                for col_idx in 0..data.ncols() {
                    let col = data.column(col_idx).to_owned();
                    let noisy_col = add_noise(&col, snr_db);
                    for (i, val) in noisy_col.iter().enumerate() {
                        result[[i, col_idx]] = *val;
                    }
                }
                result
            };
            
            save_array_2d(&noisy_data, &output)?;
            println!("Noisy data saved to: {}", output);
        }
        
        Commands::Test { output } => {
            println!("Generating test data matching Julia implementation...");
            let data = generate_test_data_matching_julia()?;
            println!("Generated data shape: ({}, {})", data.nrows(), data.ncols());
            println!("Data saved to: {}", output);
        }
        
        Commands::Legacy {
            ascii: _,
            model: _,
            tau,
            dm,
            order,
            nr_tau: _,
            data_fn,
            out_fn,
            wl,
            ws,
            select,
            ch_list,
            wl_ct: _,
            ws_ct: _,
        } => {
            println!("Running legacy DDA analysis...");
            
            // Load input data
            let data = load_data(&data_fn)?;
            
            // Convert 1-based channel indices to 0-based
            let ch_list_0based: Vec<usize> = ch_list.iter().map(|&ch| ch.saturating_sub(1)).collect();
            
            // Run selected analyses based on SELECT flags
            if select.len() != 4 {
                return Err(anyhow::anyhow!("SELECT must have exactly 4 values"));
            }
            
            // ST analysis (select[0])
            if select[0] == 1 {
                println!("Running Single Timeseries analysis...");
                if data.ncols() == 1 {
                    let data_1d = data.column(0).to_owned();
                    let st = compute_st_single(&data_1d, &tau, dm, order, wl, ws);
                    save_st_legacy(&st, &format!("{}_ST", out_fn))?;
                } else {
                    let st = compute_st_multiple(&data, &tau, dm, order, wl, ws);
                    save_st_multiple_legacy(&st, &format!("{}_ST", out_fn))?;
                }
            }
            
            // CT analysis (select[1])
            if select[1] == 1 {
                println!("Running Cross Timeseries analysis...");
                if data.ncols() < 2 {
                    return Err(anyhow::anyhow!("CT analysis requires at least 2 channels"));
                }
                
                let channel_pairs = if ch_list.is_empty() {
                    None
                } else {
                    // Convert channel list to pairs (assumes pairs are listed sequentially)
                    let mut pairs = Vec::new();
                    for chunk in ch_list_0based.chunks(2) {
                        if chunk.len() == 2 {
                            pairs.push((chunk[0], chunk[1]));
                        }
                    }
                    Some(pairs)
                };
                
                let (ct, pairs_used) = compute_ct_multiple(&data, &tau, dm, order, wl, ws, channel_pairs);
                save_ct_legacy(&ct, &pairs_used, &format!("{}_CT", out_fn))?;
            }
            
            // DE analysis (select[2]) - would implement if needed
            if select[2] == 1 {
                println!("Dynamical Ergodicity analysis not yet implemented in legacy mode");
            }
            
            println!("Legacy DDA analysis completed");
        }
    }

    Ok(())
}

fn load_data(path: &str) -> Result<Array2<f64>> {
    let path = PathBuf::from(path);
    
    // Check file extension
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("edf") | Some("EDF") => {
            Err(anyhow::anyhow!(
                "EDF files require conversion to ASCII format first.\n\
                Use: python3 edf_to_ascii.py {} -o {}\n\n\
                The Python converter supports pyedflib and mne backends for reading EDF files.",
                path.display(),
                path.with_extension("txt").display()
            ))
        }
        _ => load_ascii_data(&path),
    }
}


fn load_ascii_data(path: &PathBuf) -> Result<Array2<f64>> {
    let contents = fs::read_to_string(path)?;
    let mut result = Vec::new();
    
    for line in contents.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let row: Vec<f64> = line
            .split_whitespace()
            .map(|s| s.parse())
            .collect::<Result<Vec<_>, _>>()?;
        result.push(row);
    }
    
    if result.is_empty() {
        return Err(anyhow::anyhow!("Empty data file"));
    }
    
    let rows = result.len();
    let cols = result[0].len();
    let flat: Vec<f64> = result.into_iter().flatten().collect();
    
    Ok(Array2::from_shape_vec((rows, cols), flat)?)
}

fn save_array_2d(array: &Array2<f64>, path: &str) -> Result<()> {
    let path = PathBuf::from(path);
    let mut content = String::new();
    
    for row in array.outer_iter() {
        let row_str: Vec<String> = row.iter().map(|x| format!("{:.16e}", x)).collect();
        content.push_str(&row_str.join(" "));
        content.push('\n');
    }
    
    fs::write(&path, content)?;
    Ok(())
}

fn save_st_legacy(st: &Array2<f64>, path: &str) -> Result<()> {
    let path = PathBuf::from(path);
    let mut content = String::new();
    
    for (window, row) in st.outer_iter().enumerate() {
        // Format: window_idx channel_idx coeff1 coeff2 coeff3 error
        content.push_str(&format!("{} 0", window));
        for val in row.iter() {
            content.push_str(&format!(" {:.16e}", val));
        }
        content.push('\n');
    }
    
    fs::write(&path, content)?;
    Ok(())
}

fn save_st_multiple_legacy(st: &ndarray::Array3<f64>, path: &str) -> Result<()> {
    let path = PathBuf::from(path);
    let mut content = String::new();
    
    for window in 0..st.dim().0 {
        for channel in 0..st.dim().2 {
            // Format: window_idx channel_idx coeff1 coeff2 coeff3 error
            content.push_str(&format!("{} {}", window, channel));
            for feat in 0..st.dim().1 {
                content.push_str(&format!(" {:.16e}", st[[window, feat, channel]]));
            }
            content.push('\n');
        }
    }
    
    fs::write(&path, content)?;
    Ok(())
}

fn save_ct_legacy(ct: &ndarray::Array3<f64>, pairs: &Array2<usize>, path: &str) -> Result<()> {
    let path = PathBuf::from(path);
    let mut content = String::new();
    
    for window in 0..ct.dim().0 {
        for pair in 0..ct.dim().2 {
            // Format: window_idx ch1 ch2 coeff1 coeff2 coeff3 error  
            // Convert back to 1-based indexing for output
            content.push_str(&format!("{} {} {}", 
                window, 
                pairs[[pair, 0]] + 1, 
                pairs[[pair, 1]] + 1));
            for feat in 0..ct.dim().1 {
                content.push_str(&format!(" {:.16e}", ct[[window, feat, pair]]));
            }
            content.push('\n');
        }
    }
    
    fs::write(&path, content)?;
    Ok(())
}
