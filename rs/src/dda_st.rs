use anyhow::Result;
use ndarray::{Array1, Array2, Array3};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::dda_functions::deriv_all;

/// Compute single timeseries DDA structure coefficients.
pub fn compute_st_single(
    data: &Array1<f64>,
    tau: &[usize],
    dm: usize,
    order: usize,
    wl: usize,
    ws: usize,
) -> Array2<f64> {
    let tm = *tau.iter().max().unwrap();
    let wn = 1 + ((data.len() - (wl + tm + 2 * dm - 1)) / ws);

    let mut st = Array2::from_elem((wn, 4), f64::NAN);

    for wn_idx in 0..wn {
        let anf = wn_idx * ws;
        let ende = anf + wl + tm + 2 * dm - 1;

        let window_data = data.slice(s![anf..=ende]).to_owned();
        let ddata = deriv_all(&window_data, dm, order, 1.0);
        let window_data = window_data.slice(s![dm..(window_data.len() - dm)]).to_owned();

        let std = calculate_std(&window_data);
        let mean = window_data.mean().unwrap();
        let data_norm = (window_data.clone() - mean) / std;
        let ddata_norm = ddata / std;

        // Build design matrix M with delay coordinates
        let len_data = data_norm.len();
        let mut m = Array2::zeros((len_data - tm, 3));
        
        for i in 0..(len_data - tm) {
            m[[i, 0]] = data_norm[tm - tau[0] + i];
            m[[i, 1]] = data_norm[tm - tau[1] + i];
            m[[i, 2]] = data_norm[tm - tau[0] + i].powi(3);
        }

        // Slice derivative data AFTER matrix construction
        let ddata_sliced = ddata_norm.slice(s![tm..]).to_owned();

        // Solve for coefficients using normal equations
        let coeffs = solve_least_squares(&m, &ddata_sliced);
        
        for i in 0..3 {
            st[[wn_idx, i]] = coeffs[i];
        }

        // Compute residual error
        let prediction = m.dot(&coeffs);
        let residual = &ddata_sliced - &prediction;
        st[[wn_idx, 3]] = (residual.mapv(|x| x * x).mean().unwrap()).sqrt();
    }

    st
}

/// Compute single timeseries DDA for multiple timeseries.
pub fn compute_st_multiple(
    y: &Array2<f64>,
    tau: &[usize],
    dm: usize,
    order: usize,
    wl: usize,
    ws: usize,
) -> Array3<f64> {
    let tm = *tau.iter().max().unwrap();
    let wn = 1 + ((y.nrows() - (wl + tm + 2 * dm - 1)) / ws);
    let n_channels = y.ncols();

    let mut st = Array3::from_elem((wn, 4, n_channels), f64::NAN);

    for n_y in 0..n_channels {
        for wn_idx in 0..wn {
            let anf = wn_idx * ws;
            let ende = anf + wl + tm + 2 * dm - 1;

            let data = y.slice(s![anf..=ende, n_y]).to_owned();
            let ddata = deriv_all(&data, dm, order, 1.0);
            let data = data.slice(s![dm..(data.len() - dm)]).to_owned();

            let std = calculate_std(&data);
            let mean = data.mean().unwrap();
            let data_norm = (data.clone() - mean) / std;
            let ddata_norm = ddata / std;

            // Build design matrix M with delay coordinates
            let len_data = data_norm.len();
            let mut m = Array2::zeros((len_data - tm, 3));
            
            for i in 0..(len_data - tm) {
                m[[i, 0]] = data_norm[tm - tau[0] + i];
                m[[i, 1]] = data_norm[tm - tau[1] + i];
                m[[i, 2]] = data_norm[tm - tau[0] + i].powi(3);
            }

            // Slice derivative data AFTER matrix construction
            let ddata_sliced = ddata_norm.slice(s![tm..]).to_owned();

            // Solve for coefficients
            let coeffs = solve_least_squares(&m, &ddata_sliced);
            
            for i in 0..3 {
                st[[wn_idx, i, n_y]] = coeffs[i];
            }

            // Compute residual error
            let prediction = m.dot(&coeffs);
            let residual = &ddata_sliced - &prediction;
            st[[wn_idx, 3, n_y]] = (residual.mapv(|x| x * x).mean().unwrap()).sqrt();
        }
    }

    st
}

/// Run external DDA executable for ST analysis.
pub fn run_dda_st_external(
    fn_data: &str,
    fn_dda: &str,
    model: &[i32],
    tau: &[usize],
    dm: usize,
    dda_order: usize,
    nr_delays: usize,
    wl: usize,
    ws: usize,
    ch_list: Option<&[usize]>,
) -> Result<(String, Array2<f64>)> {
    // Platform-specific executable handling - check current dir first, then parent
    let executable = if cfg!(windows) {
        let exe_path = Path::new("run_DDA_AsciiEdf.exe");
        let parent_exe = Path::new("../run_DDA_AsciiEdf");
        
        if exe_path.exists() {
            exe_path
        } else if parent_exe.exists() {
            if !exe_path.exists() {
                std::fs::copy(parent_exe, &exe_path)?;
            }
            exe_path
        } else {
            return Err(anyhow::anyhow!("run_DDA_AsciiEdf executable not found"));
        }
    } else {
        let local_exe = Path::new("run_DDA_AsciiEdf");
        let parent_exe = Path::new("../run_DDA_AsciiEdf");
        
        if local_exe.exists() {
            local_exe
        } else if parent_exe.exists() {
            parent_exe
        } else {
            return Err(anyhow::anyhow!("run_DDA_AsciiEdf executable not found"));
        }
    };

    // Build command
    let mut cmd = Command::new(executable);
    cmd.arg("-ASCII");
    
    cmd.arg("-MODEL");
    for m in model {
        cmd.arg(m.to_string());
    }
    
    cmd.arg("-TAU");
    for t in tau {
        cmd.arg(t.to_string());
    }
    
    cmd.args(&["-dm", &dm.to_string()]);
    cmd.args(&["-order", &dda_order.to_string()]);
    cmd.args(&["-nr_tau", &nr_delays.to_string()]);
    cmd.args(&["-DATA_FN", fn_data]);
    cmd.args(&["-OUT_FN", fn_dda]);
    cmd.args(&["-WL", &wl.to_string()]);
    cmd.args(&["-WS", &ws.to_string()]);
    cmd.args(&["-SELECT", "1", "0", "0", "0"]); // ST only

    if let Some(channels) = ch_list {
        cmd.arg("-CH_list");
        for ch in channels {
            cmd.arg(ch.to_string());
        }
    }

    // Execute command
    let _output = cmd.output()?;
    
    let cmd_str = format!("{:?}", cmd);

    // Load results
    let st_file = PathBuf::from(format!("{}_ST", fn_dda));
    let contents = std::fs::read_to_string(&st_file)?;
    
    let mut results = Vec::new();
    for line in contents.lines() {
        let row: Vec<f64> = line
            .split_whitespace()
            .skip(2) // Skip first 2 columns
            .map(|s| s.parse().unwrap())
            .collect();
        results.push(row);
    }
    
    let rows = results.len();
    let cols = if rows > 0 { results[0].len() } else { 0 };
    let flat: Vec<f64> = results.into_iter().flatten().collect();
    let st_results = Array2::from_shape_vec((rows, cols), flat)?;

    Ok((cmd_str, st_results))
}

// Helper functions
fn calculate_std(data: &Array1<f64>) -> f64 {
    let mean = data.mean().unwrap();
    let variance = data.mapv(|x| (x - mean).powi(2)).sum() / (data.len() - 1) as f64;
    variance.sqrt()
}

fn solve_least_squares(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let ata = a.t().dot(a);
    let atb = a.t().dot(b);
    
    // Use Gaussian elimination with partial pivoting
    let n = ata.nrows();
    let mut aug = Array2::zeros((n, n + 1));
    
    // Copy into augmented matrix
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = ata[[i, j]];
        }
        aug[[i, n]] = atb[i];
    }
    
    // Forward elimination with partial pivoting
    for k in 0..n {
        // Find pivot row
        let mut max_row = k;
        for i in (k+1)..n {
            if aug[[i, k]].abs() > aug[[max_row, k]].abs() {
                max_row = i;
            }
        }
        
        // Swap rows if needed
        if max_row != k {
            for j in 0..(n+1) {
                let temp = aug[[k, j]];
                aug[[k, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }
        
        // Eliminate
        for i in (k+1)..n {
            let factor = aug[[i, k]] / aug[[k, k]];
            for j in k..(n+1) {
                aug[[i, j]] -= factor * aug[[k, j]];
            }
        }
    }
    
    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        x[i] = aug[[i, n]];
        for j in (i+1)..n {
            x[i] -= aug[[i, j]] * x[j];
        }
        x[i] /= aug[[i, i]];
    }
    
    x
}