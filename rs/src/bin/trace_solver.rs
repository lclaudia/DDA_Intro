use dda::dda_functions::deriv_all;
use ndarray::{Array1, Array2, s};
// use ndarray_linalg::Solve;
use std::fs;

fn main() {
    // Load test data
    let data_str = fs::read_to_string("trace_st_input.txt").expect("Failed to read trace_st_input.txt");
    let data: Vec<f64> = data_str.lines()
        .map(|line| line.trim().parse().unwrap())
        .collect();
    let data = Array1::from(data);
    
    // Parameters from Python
    let tau = vec![3, 15];
    let dm = 3;
    let order = 2;
    let wl = 70;
    
    // Calculate exact same window as Python
    let tm = *tau.iter().max().unwrap();
    let anf = 0;
    let ende = anf + wl + tm + 2 * dm - 1;
    
    println!("Data length: {}", data.len());
    println!("tm: {}, wl: {}, dm: {}", tm, wl, dm);
    println!("Required ende: {}", ende);
    
    if ende >= data.len() {
        println!("Error: Not enough data. Need at least {} points, have {}", ende + 1, data.len());
        return;
    }
    
    let window_data = data.slice(s![anf..=ende]).to_owned();
    let ddata = deriv_all(&window_data, dm, order, 1.0);
    let window_data = window_data.slice(s![dm..(window_data.len() - dm)]).to_owned();
    
    let std_val = calculate_std(&window_data);
    let mean = window_data.mean().unwrap();
    let data_norm = (&window_data - mean) / std_val;
    let ddata_norm = ddata / std_val;
    
    // Build design matrix
    let len_data = data_norm.len();
    let mut m = Array2::zeros((len_data - tm, 3));
    
    for i in 0..(len_data - tm) {
        m[[i, 0]] = data_norm[tm - tau[0] + i];
        m[[i, 1]] = data_norm[tm - tau[1] + i]; 
        m[[i, 2]] = data_norm[tm - tau[0] + i].powi(3);
    }
    
    let ddata_sliced = ddata_norm.slice(s![tm..]).to_owned();
    
    println!("Matrix M shape: {:?}", m.dim());
    println!("Vector b shape: {:?}", ddata_sliced.len());
    
    // Step 1: Form normal equations A^T A and A^T b
    let ata = m.t().dot(&m);
    let atb = m.t().dot(&ddata_sliced);
    
    println!("A^T A matrix:");
    for i in 0..3 {
        for j in 0..3 {
            println!("  [{},{}] = {:.16e}", i, j, ata[[i, j]]);
        }
    }
    
    println!("A^T b vector:");
    for i in 0..3 {
        println!("  [{}] = {:.16e}", i, atb[i]);
    }
    
    // Step 2: Manual solver
    let solution_manual = solve_manual(&ata, &atb);
    println!("Manual solution:");
    for i in 0..3 {
        println!("  [{}] = {:.16e}", i, solution_manual[i]);
    }
    
    
    // Write matrices to files for Python comparison
    write_matrix_to_file(&ata, "rust_ata.txt");
    write_vector_to_file(&atb, "rust_atb.txt");
    write_vector_to_file(&solution_manual, "rust_solution.txt");
}

fn calculate_std(data: &Array1<f64>) -> f64 {
    let mean = data.mean().unwrap();
    let variance = data.mapv(|x| (x - mean).powi(2)).sum() / (data.len() - 1) as f64;
    variance.sqrt()
}

fn solve_manual(ata: &Array2<f64>, atb: &Array1<f64>) -> Array1<f64> {
    // Manual Gaussian elimination with partial pivoting
    let n = ata.nrows();
    let mut aug = Array2::zeros((n, n + 1));
    
    // Copy A into augmented matrix
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = ata[[i, j]];
        }
        aug[[i, n]] = atb[i];
    }
    
    println!("Augmented matrix before elimination:");
    for i in 0..n {
        for j in 0..(n+1) {
            print!("{:.16e} ", aug[[i, j]]);
        }
        println!();
    }
    
    // Forward elimination
    for k in 0..n {
        // Partial pivoting
        let mut max_row = k;
        for i in (k+1)..n {
            if aug[[i, k]].abs() > aug[[max_row, k]].abs() {
                max_row = i;
            }
        }
        
        if max_row != k {
            for j in 0..(n+1) {
                let temp = aug[[k, j]];
                aug[[k, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }
        
        println!("After pivot step {}, pivot = {:.16e}", k, aug[[k, k]]);
        
        // Eliminate below
        for i in (k+1)..n {
            let factor = aug[[i, k]] / aug[[k, k]];
            println!("  Eliminating row {} with factor {:.16e}", i, factor);
            
            for j in k..(n+1) {
                aug[[i, j]] -= factor * aug[[k, j]];
            }
        }
    }
    
    println!("After forward elimination:");
    for i in 0..n {
        for j in 0..(n+1) {
            print!("{:.16e} ", aug[[i, j]]);
        }
        println!();
    }
    
    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        x[i] = aug[[i, n]];
        for j in (i+1)..n {
            x[i] -= aug[[i, j]] * x[j];
        }
        x[i] /= aug[[i, i]];
        println!("Back substitution step {}: x[{}] = {:.16e}", i, i, x[i]);
    }
    
    x
}

fn write_matrix_to_file(matrix: &Array2<f64>, filename: &str) {
    let mut content = String::new();
    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            content.push_str(&format!("{:.16e}", matrix[[i, j]]));
            if j < matrix.ncols() - 1 {
                content.push(' ');
            }
        }
        content.push('\n');
    }
    fs::write(filename, content).expect("Failed to write matrix file");
}

fn write_vector_to_file(vector: &Array1<f64>, filename: &str) {
    let mut content = String::new();
    for i in 0..vector.len() {
        content.push_str(&format!("{:.16e}\n", vector[i]));
    }
    fs::write(filename, content).expect("Failed to write vector file");
}