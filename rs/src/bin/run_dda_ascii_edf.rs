use anyhow::Result;
use clap::Parser;
use ndarray::Array2;
use std::fs;
use std::path::PathBuf;

use dda::{compute_st_multiple, compute_ct_multiple};

#[derive(Parser)]
#[command(name = "run_DDA_AsciiEdf")]
#[command(about = "Legacy DDA executable compatible with Python/Julia scripts", long_about = None)]
struct Args {
    /// ASCII format flag
    #[arg(short = 'A', long = "ASCII", action = clap::ArgAction::SetTrue)]
    ascii: bool,
    
    /// Model specification
    #[arg(long = "MODEL", num_args = 1.., value_delimiter = ' ', require_equals = true)]
    model: Vec<i32>,
    
    /// Delay values  
    #[arg(long = "TAU", num_args = 1.., value_delimiter = ' ', require_equals = true)]
    tau: Vec<usize>,
    
    /// Derivative method parameter
    #[arg(long)]
    dm: usize,
    
    /// DDA order
    #[arg(long)]
    order: usize,
    
    /// Number of delays
    #[arg(long = "nr_tau")]
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
    #[arg(long = "SELECT", num_args = 4)]
    select: Vec<u8>,
    
    /// Channel list for cross-timeseries analysis
    #[arg(long = "CH_list", num_args = 0.., value_delimiter = ' ', allow_hyphen_values = true)]
    ch_list: Vec<usize>,
    
    /// Window length for CT analysis
    #[arg(long = "WL_CT")]
    wl_ct: Option<usize>,
    
    /// Window shift for CT analysis  
    #[arg(long = "WS_CT")]
    ws_ct: Option<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    // Load input data
    let data = load_data(&args.data_fn)?;
    
    // Convert 1-based channel indices to 0-based
    let ch_list_0based: Vec<usize> = args.ch_list.iter().map(|&ch| ch.saturating_sub(1)).collect();
    
    // Run selected analyses based on SELECT flags
    if args.select.len() != 4 {
        return Err(anyhow::anyhow!("SELECT must have exactly 4 values"));
    }
    
    // ST analysis (select[0])
    if args.select[0] == 1 {
        if data.ncols() == 1 {
            let data_1d = data.column(0).to_owned();
            let st = dda::compute_st_single(&data_1d, &args.tau, args.dm, args.order, args.wl, args.ws);
            save_st_legacy(&st, &format!("{}_ST", args.out_fn))?;
        } else {
            let st = compute_st_multiple(&data, &args.tau, args.dm, args.order, args.wl, args.ws);
            save_st_multiple_legacy(&st, &format!("{}_ST", args.out_fn))?;
        }
    }
    
    // CT analysis (select[1])
    if args.select[1] == 1 {
        if data.ncols() < 2 {
            return Err(anyhow::anyhow!("CT analysis requires at least 2 channels"));
        }
        
        let channel_pairs = if args.ch_list.is_empty() {
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
        
        let (ct, pairs_used) = compute_ct_multiple(&data, &args.tau, args.dm, args.order, args.wl, args.ws, channel_pairs);
        save_ct_legacy(&ct, &pairs_used, &format!("{}_CT", args.out_fn))?;
    }
    
    // DE analysis (select[2]) - would implement if needed
    if args.select[2] == 1 {
        eprintln!("Warning: Dynamical Ergodicity analysis not yet implemented");
    }
    
    Ok(())
}

fn load_data(path: &str) -> Result<Array2<f64>> {
    let path = PathBuf::from(path);
    let contents = fs::read_to_string(&path)?;
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