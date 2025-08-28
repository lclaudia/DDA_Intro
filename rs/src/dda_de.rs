use anyhow::Result;
use ndarray::{Array2, Array3, Axis};
use std::collections::HashMap;

use crate::dda_ct::compute_ct_multiple;
use crate::dda_st::compute_st_multiple;

/// Compute dynamical ergodicity matrix from ST and CT results.
///
/// The dynamical ergodicity measure quantifies how well the coupling
/// between two timeseries can be explained by their individual structures.
pub fn compute_dynamical_ergodicity(
    st: &Array3<f64>,
    ct: &Array3<f64>,
    channel_pairs: &Array2<usize>,
) -> Array2<f64> {
    let n_channels = st.dim().2;
    
    // Compute mean errors over windows
    let st_mean = st.slice(s![.., 3, ..]).mean_axis(Axis(0)).unwrap(); // Mean over windows, last column (errors)
    let ct_mean = ct.slice(s![.., 3, ..]).mean_axis(Axis(0)).unwrap(); // Mean over windows, last column (errors)
    
    // Initialize ergodicity matrix
    let mut e = Array2::from_elem((n_channels, n_channels), f64::NAN);
    
    // Fill the matrix with ergodicity values
    for (n_pair, row) in channel_pairs.outer_iter().enumerate() {
        let ch1 = row[0];
        let ch2 = row[1];
        
        // Ergodicity measure: |mean([st_i, st_j])/ct_ij - 1|
        let mean_st = (st_mean[ch1] + st_mean[ch2]) / 2.0;
        e[[ch1, ch2]] = (mean_st / ct_mean[n_pair] - 1.0).abs();
        e[[ch2, ch1]] = e[[ch1, ch2]]; // Symmetric matrix
    }
    
    e
}

/// Compute statistics about the ergodicity matrix.
pub fn analyze_ergodicity_statistics(e: &Array2<f64>, threshold: f64) -> ErgodicityStat {
    // Extract upper triangle (excluding diagonal)
    let n = e.nrows();
    let mut values = Vec::new();
    let mut ergodic_pairs = Vec::new();
    
    for i in 0..n {
        for j in (i + 1)..n {
            let val = e[[i, j]];
            if !val.is_nan() {
                values.push(val);
                if val < threshold {
                    ergodic_pairs.push((i, j));
                }
            }
        }
    }
    
    let mean = if values.is_empty() {
        f64::NAN
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    };
    
    let std = if values.len() <= 1 {
        f64::NAN
    } else {
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt()
    };
    
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    
    let total_pairs = values.len();
    let n_ergodic = ergodic_pairs.len();
    let ergodic_fraction = if total_pairs > 0 {
        n_ergodic as f64 / total_pairs as f64
    } else {
        0.0
    };
    
    ErgodicityStat {
        mean,
        std,
        min: if min.is_infinite() { f64::NAN } else { min },
        max: if max.is_infinite() { f64::NAN } else { max },
        n_ergodic,
        ergodic_pairs,
        total_pairs,
        ergodic_fraction,
    }
}

/// Compare computed ergodicity matrix with external results.
pub fn compare_with_external_de(
    e_computed: &Array2<f64>,
    e_external: &Array2<f64>,
    tolerance: f64,
) -> (f64, bool) {
    // Compute difference only for non-NaN values
    let mut diffs = Vec::new();
    
    for i in 0..e_computed.nrows() {
        for j in 0..e_computed.ncols() {
            let val_computed = e_computed[[i, j]];
            let val_external = e_external[[i, j]];
            
            if !val_computed.is_nan() && !val_external.is_nan() {
                diffs.push((val_computed - val_external).abs());
            }
        }
    }
    
    if diffs.is_empty() {
        return (f64::NAN, false);
    }
    
    let mean_error = diffs.iter().sum::<f64>() / diffs.len() as f64;
    let is_within_tolerance = diffs.iter().all(|&diff| diff < tolerance);
    
    (mean_error, is_within_tolerance)
}

/// Run complete dynamical ergodicity analysis on data.
pub fn run_full_de_analysis(
    y: &Array2<f64>,
    tau: &[usize],
    dm: usize,
    order: usize,
    wl: usize,
    ws: usize,
) -> Result<(Array2<f64>, ErgodicityStat)> {
    // Compute ST for all channels
    let st = compute_st_multiple(y, tau, dm, order, wl, ws);
    
    // Compute CT for all channel pairs
    let (ct, channel_pairs) = compute_ct_multiple(y, tau, dm, order, wl, ws, None);
    
    // Compute ergodicity matrix
    let e = compute_dynamical_ergodicity(&st, &ct, &channel_pairs);
    
    // Compute statistics
    let stats = analyze_ergodicity_statistics(&e, 0.1);
    
    Ok((e, stats))
}

/// Structure to hold ergodicity statistics
#[derive(Debug, Clone)]
pub struct ErgodicityStat {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub n_ergodic: usize,
    pub ergodic_pairs: Vec<(usize, usize)>,
    pub total_pairs: usize,
    pub ergodic_fraction: f64,
}

impl ErgodicityStat {
    pub fn to_hashmap(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("mean".to_string(), self.mean.to_string());
        map.insert("std".to_string(), self.std.to_string());
        map.insert("min".to_string(), self.min.to_string());
        map.insert("max".to_string(), self.max.to_string());
        map.insert("n_ergodic".to_string(), self.n_ergodic.to_string());
        map.insert("total_pairs".to_string(), self.total_pairs.to_string());
        map.insert("ergodic_fraction".to_string(), self.ergodic_fraction.to_string());
        
        // Format ergodic pairs
        let pairs_str = self.ergodic_pairs
            .iter()
            .map(|(i, j)| format!("({},{})", i, j))
            .collect::<Vec<_>>()
            .join(", ");
        map.insert("ergodic_pairs".to_string(), pairs_str);
        
        map
    }
}

