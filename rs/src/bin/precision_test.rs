use ndarray::{s, Array1, Array2};
use std::fs::File;
use std::io::{Write, BufRead, BufReader};

fn read_array_from_file(filename: &str) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let mut values = Vec::new();
    
    for line in reader.lines() {
        let line = line?;
        if !line.trim().is_empty() && !line.starts_with('#') {
            values.push(line.trim().parse::<f64>()?);
        }
    }
    
    Ok(Array1::from_vec(values))
}

fn test_derivative_exact() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(60));
    println!("RUST DERIVATIVE CALCULATION TRACE");
    println!("{}", "=".repeat(60));
    
    // Test data matching Python exactly
    let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let dm = 2;
    let order = 2;
    let dt = 1.0;
    
    println!("Input data: {:?}", data);
    println!("dm={}, order={}, dt={}", dm, order, dt);
    
    // Exact replication of Python's algorithm
    let t: Vec<usize> = (dm..data.len() - dm).collect();
    println!("\nIndex range t: {:?}", t);
    
    let mut ddata = Array1::zeros(t.len());
    
    // Match Python's loop exactly
    for n1 in 1..=dm {
        println!("\nn1 = {}:", n1);
        for (i, &ti) in t.iter().enumerate() {
            let forward = data[ti + n1];
            let backward = data[ti - n1];
            let contrib = (forward - backward) / n1 as f64;
            ddata[i] += contrib;
            
            if i < 3 {
                println!("  i={}, ti={}: data[{}]-data[{}] = {:.16e}-{:.16e} = {:.16e}, contrib={:.16e}", 
                    i, ti, ti+n1, ti-n1, forward, backward, forward-backward, contrib);
            }
        }
    }
    
    // Final scaling - exactly as Python
    ddata /= dm as f64 / dt;
    println!("\nFinal scaling by dm/dt = {}", dm as f64 / dt);
    println!("Final ddata: {:?}", ddata);
    
    // Save result
    let mut file = File::create("rust_deriv_exact.txt")?;
    for val in ddata.iter() {
        writeln!(file, "{:.16e}", val)?;
    }
    
    // Load Python result
    let python_deriv = read_array_from_file("python_deriv_trace.txt")?;
    println!("\nPython result: {:?}", python_deriv);
    
    // Compare
    let diff: f64 = ddata.iter()
        .zip(python_deriv.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    
    println!("\nMax difference: {:.2e}", diff);
    if diff < 1e-15 {
        println!("✅ ACHIEVED MACHINE PRECISION!");
    }
    
    Ok(())
}

fn test_st_exact() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(60));
    println!("RUST ST CALCULATION TRACE");
    println!("{}", "=".repeat(60));
    
    // Load test data
    let data = read_array_from_file("trace_st_input.txt")?;
    println!("Loaded data: {} points", data.len());
    
    let tau = vec![2, 5];
    let dm = 2;
    let wl = 50;
    
    // First window
    let tm = *tau.iter().max().unwrap();
    let anf = 0;
    let ende = anf + wl + tm + 2 * dm - 1;
    
    println!("Window 0: anf={}, ende={}", anf, ende);
    
    let window_data = data.slice(s![anf..=ende]).to_owned();
    println!("Window data shape: {}", window_data.len());
    
    // Calculate derivative using exact same algorithm as Python
    let t: Vec<usize> = (dm..window_data.len() - dm).collect();
    let mut ddata = Array1::zeros(t.len());
    
    // Order 3 derivative (matching Python)
    let mut d = 0;
    for n1 in 1..=dm {
        for n2 in (n1 + 1)..=dm {
            d += 1;
            for (i, &ti) in t.iter().enumerate() {
                let n1_3 = (n1 as f64).powi(3);
                let n2_3 = (n2 as f64).powi(3);
                ddata[i] -= ((window_data[ti - n2] - window_data[ti + n2]) * n1_3
                    - (window_data[ti - n1] - window_data[ti + n1]) * n2_3)
                    / (n1_3 * n2 as f64 - n1 as f64 * n2_3);
            }
        }
    }
    ddata /= d as f64;  // dt = 1.0
    
    // Extract data after derivative
    let window_data_trimmed = window_data.slice(s![dm..(window_data.len() - dm)]).to_owned();
    
    // Standardize exactly as Python
    let mean = window_data_trimmed.mean().unwrap();
    let variance = window_data_trimmed.mapv(|x| (x - mean).powi(2)).sum() / (window_data_trimmed.len() - 1) as f64;
    let std = variance.sqrt();
    
    println!("Mean={:.16e}, STD={:.16e}", mean, std);
    
    let data_norm = (window_data_trimmed.clone() - mean) / std;
    let ddata_norm = ddata / std;
    
    println!("First 5 DATA values: {:?}", data_norm.slice(s![0..5]));
    println!("First 5 dDATA values: {:?}", ddata_norm.slice(s![0..5]));
    
    // Build design matrix exactly as Python
    let len_data = data_norm.len();
    let mut m = Array2::zeros((len_data - tm, 3));
    
    for i in 0..(len_data - tm) {
        m[[i, 0]] = data_norm[tm - tau[0] + i];
        m[[i, 1]] = data_norm[tm - tau[1] + i];
        m[[i, 2]] = data_norm[tm - tau[0] + i].powi(3);
    }
    
    println!("\nMatrix M shape: {:?}", m.shape());
    println!("M[0,:] = [{:.16e}, {:.16e}, {:.16e}]", m[[0,0]], m[[0,1]], m[[0,2]]);
    println!("M[1,:] = [{:.16e}, {:.16e}, {:.16e}]", m[[1,0]], m[[1,1]], m[[1,2]]);
    
    // Target vector
    let ddata_sliced = ddata_norm.slice(s![tm..]).to_owned();
    println!("\ndDATA_sliced shape: {}", ddata_sliced.len());
    println!("First 5 values: {:?}", ddata_sliced.slice(s![0..5]));
    
    // Normal equations
    let mtm = m.t().dot(&m);
    let mtd = m.t().dot(&ddata_sliced);
    
    println!("\nM'M matrix:");
    for i in 0..3 {
        println!("  [{:.16e}, {:.16e}, {:.16e}]", mtm[[i,0]], mtm[[i,1]], mtm[[i,2]]);
    }
    
    println!("\nM'd vector:");
    for i in 0..3 {
        println!("  {:.16e}", mtd[i]);
    }
    
    // Save matrices for comparison
    let mut file = File::create("rust_MtM.txt")?;
    for i in 0..3 {
        for j in 0..3 {
            writeln!(file, "{:.16e}", mtm[[i,j]])?;
        }
    }
    
    let mut file = File::create("rust_Mtd.txt")?;
    for i in 0..3 {
        writeln!(file, "{:.16e}", mtd[i])?;
    }
    
    // Load Python matrices
    let python_mtm = read_array_from_file("trace_MtM.txt")?;
    let python_mtd = read_array_from_file("trace_Mtd.txt")?;
    
    // Compare matrices
    println!("\nMatrix comparison:");
    let mut max_diff = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            let py_val = python_mtm[i * 3 + j];
            let rs_val = mtm[[i, j]];
            let diff = (py_val - rs_val).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > 1e-15 {
                println!("  MtM[{},{}]: Python={:.16e}, Rust={:.16e}, diff={:.2e}", 
                    i, j, py_val, rs_val, diff);
            }
        }
    }
    
    for i in 0..3 {
        let py_val = python_mtd[i];
        let rs_val = mtd[i];
        let diff = (py_val - rs_val).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 1e-15 {
            println!("  Mtd[{}]: Python={:.16e}, Rust={:.16e}, diff={:.2e}", 
                i, py_val, rs_val, diff);
        }
    }
    
    if max_diff < 1e-15 {
        println!("\n✅ Matrices match at machine precision!");
    } else {
        println!("\n❌ Matrix difference: {:.2e}", max_diff);
    }
    
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("RUST PRECISION TEST - Achieving e-16");
    println!("{}", "=".repeat(60));
    
    test_derivative_exact()?;
    test_st_exact()?;
    
    Ok(())
}