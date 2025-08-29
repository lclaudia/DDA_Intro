use anyhow::Result;
use itertools::Itertools;
use ndarray::{Array1, Array2, Array3, concatenate, Axis};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::dda_functions::deriv_all;

/// Compute cross-timeseries DDA outputs for a pair of signals.
pub fn compute_ct_pair(
    data1: &Array1<f64>,
    data2: &Array1<f64>,
    tau: &[usize],
    dm: usize,
    order: usize,
    wl: usize,
    ws: usize,
) -> Array2<f64> {
    let tm = *tau.iter().max().unwrap();
    let wn = 1 + ((data1.len() - (wl + tm + 2 * dm - 1)) / ws);

    let mut ct = Array2::from_elem((wn, 4), f64::NAN);

    for wn_idx in 0..wn {
        let anf = wn_idx * ws;
        let ende = anf + wl + tm + 2 * dm - 1;

        // Process first timeseries
        let window_data1 = data1.slice(s![anf..=ende]).to_owned();
        let ddata1 = deriv_all(&window_data1, dm, order, 1.0);
        let window_data1 = window_data1.slice(s![dm..(window_data1.len() - dm)]).to_owned();

        let std1 = calculate_std(&window_data1);
        let mean1 = window_data1.mean().unwrap();
        let data1_norm = (window_data1 - mean1) / std1;
        let ddata1_norm = ddata1 / std1;

        // Process second timeseries
        let window_data2 = data2.slice(s![anf..=ende]).to_owned();
        let ddata2 = deriv_all(&window_data2, dm, order, 1.0);
        let window_data2 = window_data2.slice(s![dm..(window_data2.len() - dm)]).to_owned();

        let std2 = calculate_std(&window_data2);
        let mean2 = window_data2.mean().unwrap();
        let data2_norm = (window_data2 - mean2) / std2;
        let ddata2_norm = ddata2 / std2;

        // Build design matrices
        let len_data = data1_norm.len();
        let mut m1 = Array2::zeros((len_data - tm, 3));
        let mut m2 = Array2::zeros((len_data - tm, 3));

        for i in 0..(len_data - tm) {
            m1[[i, 0]] = data1_norm[tm - tau[0] + i];
            m1[[i, 1]] = data1_norm[tm - tau[1] + i];
            m1[[i, 2]] = data1_norm[tm - tau[0] + i].powi(3);

            m2[[i, 0]] = data2_norm[tm - tau[0] + i];
            m2[[i, 1]] = data2_norm[tm - tau[1] + i];
            m2[[i, 2]] = data2_norm[tm - tau[0] + i].powi(3);
        }

        // Slice derivatives AFTER matrix construction
        let ddata1_sliced = ddata1_norm.slice(s![tm..]).to_owned();
        let ddata2_sliced = ddata2_norm.slice(s![tm..]).to_owned();

        // Combine matrices and data
        let m = concatenate![Axis(0), m1, m2];
        let ddata_combined = concatenate![Axis(0), ddata1_sliced, ddata2_sliced];

        // Solve for coefficients
        let coeffs = solve_least_squares(&m, &ddata_combined);
        
        for i in 0..3 {
            ct[[wn_idx, i]] = coeffs[i];
        }

        // Compute residual error
        let prediction = m.dot(&coeffs);
        let residual = &ddata_combined - &prediction;
        ct[[wn_idx, 3]] = (residual.mapv(|x| x * x).mean().unwrap()).sqrt();
    }

    ct
}

/// Compute cross-timeseries DDA for multiple channel pairs.
pub fn compute_ct_multiple(
    y: &Array2<f64>,
    tau: &[usize],
    dm: usize,
    order: usize,
    wl: usize,
    ws: usize,
    channel_pairs: Option<Vec<(usize, usize)>>,
) -> (Array3<f64>, Array2<usize>) {
    let nr_ch = y.ncols();

    let pairs = if let Some(pairs) = channel_pairs {
        pairs
    } else {
        // Generate all combinations
        (0..nr_ch).combinations(2).map(|v| (v[0], v[1])).collect()
    };

    let tm = *tau.iter().max().unwrap();
    let wn = 1 + ((y.nrows() - (wl + tm + 2 * dm - 1)) / ws);

    let mut ct = Array3::from_elem((wn, 4, pairs.len()), f64::NAN);

    for (n_list, &(ch1, ch2)) in pairs.iter().enumerate() {
        for wn_idx in 0..wn {
            let anf = wn_idx * ws;
            let ende = anf + wl + tm + 2 * dm - 1;

            // Process first channel
            let data1 = y.slice(s![anf..=ende, ch1]).to_owned();
            let ddata1 = deriv_all(&data1, dm, order, 1.0);
            let data1 = data1.slice(s![dm..(data1.len() - dm)]).to_owned();

            let std = calculate_std(&data1);
            let mean = data1.mean().unwrap();
            let data1_norm = (data1 - mean) / std;
            let ddata1_norm = ddata1 / std;

            // Process second channel
            let data2 = y.slice(s![anf..=ende, ch2]).to_owned();
            let ddata2 = deriv_all(&data2, dm, order, 1.0);
            let data2 = data2.slice(s![dm..(data2.len() - dm)]).to_owned();

            let std = calculate_std(&data2);
            let mean = data2.mean().unwrap();
            let data2_norm = (data2 - mean) / std;
            let ddata2_norm = ddata2 / std;

            // Build design matrices
            let len_data = data1_norm.len();
            let mut m1 = Array2::zeros((len_data - tm, 3));
            let mut m2 = Array2::zeros((len_data - tm, 3));

            for i in 0..(len_data - tm) {
                m1[[i, 0]] = data1_norm[tm - tau[0] + i];
                m1[[i, 1]] = data1_norm[tm - tau[1] + i];
                m1[[i, 2]] = data1_norm[tm - tau[0] + i].powi(3);

                m2[[i, 0]] = data2_norm[tm - tau[0] + i];
                m2[[i, 1]] = data2_norm[tm - tau[1] + i];
                m2[[i, 2]] = data2_norm[tm - tau[0] + i].powi(3);
            }

            // Slice derivatives AFTER matrix construction
            let ddata1_sliced = ddata1_norm.slice(s![tm..]).to_owned();
            let ddata2_sliced = ddata2_norm.slice(s![tm..]).to_owned();

            // Combine matrices and data
            let m = concatenate![Axis(0), m1, m2];
            let ddata_combined = concatenate![Axis(0), ddata1_sliced, ddata2_sliced];

            // Solve for coefficients
            let coeffs = solve_least_squares(&m, &ddata_combined);
            
            for i in 0..3 {
                ct[[wn_idx, i, n_list]] = coeffs[i];
            }

            // Compute residual error
            let prediction = m.dot(&coeffs);
            let residual = &ddata_combined - &prediction;
            ct[[wn_idx, 3, n_list]] = (residual.mapv(|x| x * x).mean().unwrap()).sqrt();
        }
    }

    // Convert pairs to Array2
    let mut list_array = Array2::zeros((pairs.len(), 2));
    for (i, &(ch1, ch2)) in pairs.iter().enumerate() {
        list_array[[i, 0]] = ch1;
        list_array[[i, 1]] = ch2;
    }

    (ct, list_array)
}

/// Run external DDA executable for CT analysis.
pub fn run_dda_ct_external(
    fn_data: &str,
    fn_dda: &str,
    model: &[i32],
    tau: &[usize],
    list: &Array2<usize>,
    dm: usize,
    dda_order: usize,
    nr_delays: usize,
    wl: usize,
    ws: usize,
    wl_ct: usize,
    ws_ct: usize,
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
    cmd.args(&["-SELECT", "0", "1", "0", "0"]); // CT only

    // Convert 0-based LIST to 1-based for external tool
    cmd.arg("-CH_list");
    for row in list.outer_iter() {
        for val in row.iter() {
            cmd.arg((val + 1).to_string());
        }
    }
    
    cmd.args(&["-WL_CT", &wl_ct.to_string()]);
    cmd.args(&["-WS_CT", &ws_ct.to_string()]);

    // Execute command
    let _output = cmd.output()?;
    
    let cmd_str = format!("{:?}", cmd);

    // Load results
    let ct_file = PathBuf::from(format!("{}_CT", fn_dda));
    let contents = std::fs::read_to_string(&ct_file)?;
    
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
    let ct_results = Array2::from_shape_vec((rows, cols), flat)?;

    Ok((cmd_str, ct_results))
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