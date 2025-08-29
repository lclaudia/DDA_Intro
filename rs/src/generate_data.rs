use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::fs;
use std::process::Command;

use crate::dda_functions::{create_mod_nr, integrate_ode_general};

/// Parameters for data generation
#[derive(Debug, Clone)]
pub struct DataGenerationParams {
    pub n_systems: usize,
    pub a_values: Vec<f64>,
    pub b_values: Vec<f64>,
    pub c: f64,
    pub dt: f64,
    pub length: usize,
    pub transient: usize,
    pub delta: usize,
    pub dimension: usize,
    pub order: usize,
    pub initial_conditions: Array1<f64>,
}

/// Generate data from coupled Roessler systems.
pub fn generate_roessler_data(
    n_systems: usize,
    a_values: Option<Vec<f64>>,
    b_values: Option<Vec<f64>>,
    c: f64,
    dt: f64,
    length: usize,
    transient: usize,
    delta: usize,
    output_file: Option<&str>,
    seed: Option<u64>,
) -> Result<(Array2<f64>, DataGenerationParams)> {
    // Define Roessler system structure
    let ros = Array2::from_shape_vec(
        (7, 3),
        vec![
            0, 0, 2,  // dx/dt terms
            0, 0, 3,
            1, 0, 1,  // dy/dt terms
            1, 0, 2,
            2, 0, 0,  // dz/dt terms
            2, 0, 3,
            2, 1, 3,
        ],
    )?;

    // Create model encoding
    let (mod_nr, dim, ode_order, _p) = create_mod_nr(&ros, n_systems);

    // Handle parameter values
    let a_values = a_values.unwrap_or_else(|| vec![0.21; n_systems]);
    let b_values = b_values.unwrap_or_else(|| {
        (0..n_systems).map(|i| 0.2 + i as f64 * 0.01).collect()
    });

    // Create parameter array
    let mut mod_par = Vec::new();
    for i in 0..n_systems {
        let params = vec![
            -1.0, -1.0, 1.0, a_values[i], b_values[i], -c, 1.0
        ];
        for p in params {
            mod_par.push(p);
        }
    }

    // Generate initial conditions
    let x0 = if let Some(seed) = seed {
        let mut rng = StdRng::seed_from_u64(seed);
        Array1::from_vec(
            (0..dim * n_systems).map(|_| rng.gen::<f64>()).collect()
        )
    } else {
        let mut rng = rand::thread_rng();
        Array1::from_vec(
            (0..dim * n_systems).map(|_| rng.gen::<f64>()).collect()
        )
    };

    // Channel list (all coordinates)
    let ch_list: Vec<usize> = (1..=dim * n_systems).collect();

    // Integrate ODE
    let data = if let Some(filename) = output_file {
        integrate_ode_general(
            &mod_nr.to_vec().iter().map(|&x| x as i32).collect::<Vec<_>>(),
            &mod_par,
            dt,
            length,
            dim * n_systems,
            ode_order,
            &x0.to_vec(),
            filename,
            &ch_list,
            delta,
            Some(transient),
        )?;
        
        // Load from file
        let contents = fs::read_to_string(filename)?;
        let mut result = Vec::new();
        for line in contents.lines() {
            let row: Vec<f64> = line
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect();
            result.push(row);
        }
        let rows = result.len();
        let cols = if rows > 0 { result[0].len() } else { 0 };
        let flat: Vec<f64> = result.into_iter().flatten().collect();
        Some(Array2::from_shape_vec((rows, cols), flat)?)
    } else {
        integrate_ode_general(
            &mod_nr.to_vec().iter().map(|&x| x as i32).collect::<Vec<_>>(),
            &mod_par,
            dt,
            length,
            dim * n_systems,
            ode_order,
            &x0.to_vec(),
            "",
            &ch_list,
            delta,
            Some(transient),
        )?
    };

    let data = data.ok_or_else(|| anyhow::anyhow!("Failed to generate data"))?;

    // Return data and parameters
    let params = DataGenerationParams {
        n_systems,
        a_values,
        b_values,
        c,
        dt,
        length,
        transient,
        delta,
        dimension: dim,
        order: ode_order,
        initial_conditions: x0,
    };

    Ok((data, params))
}

/// Generate data from custom coupled ODE systems.
pub fn generate_custom_system_data(
    system_spec: &Array2<i32>,
    n_systems: usize,
    parameters: &[f64],
    dt: f64,
    length: usize,
    transient: usize,
    delta: usize,
    output_file: Option<&str>,
    seed: Option<u64>,
) -> Result<(Array2<f64>, DataGenerationParams)> {
    // Create model encoding
    let (mod_nr, dim, ode_order, _p) = create_mod_nr(system_spec, n_systems);

    // Generate initial conditions
    let x0 = if let Some(seed) = seed {
        let mut rng = StdRng::seed_from_u64(seed);
        Array1::from_vec(
            (0..dim * n_systems).map(|_| rng.gen::<f64>()).collect()
        )
    } else {
        let mut rng = rand::thread_rng();
        Array1::from_vec(
            (0..dim * n_systems).map(|_| rng.gen::<f64>()).collect()
        )
    };

    // Channel list (all coordinates)
    let ch_list: Vec<usize> = (1..=dim * n_systems).collect();

    // Integrate ODE
    let data = if let Some(filename) = output_file {
        integrate_ode_general(
            &mod_nr.to_vec().iter().map(|&x| x as i32).collect::<Vec<_>>(),
            parameters,
            dt,
            length,
            dim * n_systems,
            ode_order,
            &x0.to_vec(),
            filename,
            &ch_list,
            delta,
            Some(transient),
        )?;
        
        // Load from file
        let contents = fs::read_to_string(filename)?;
        let mut result = Vec::new();
        for line in contents.lines() {
            let row: Vec<f64> = line
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect();
            result.push(row);
        }
        let rows = result.len();
        let cols = if rows > 0 { result[0].len() } else { 0 };
        let flat: Vec<f64> = result.into_iter().flatten().collect();
        Some(Array2::from_shape_vec((rows, cols), flat)?)
    } else {
        integrate_ode_general(
            &mod_nr.to_vec().iter().map(|&x| x as i32).collect::<Vec<_>>(),
            parameters,
            dt,
            length,
            dim * n_systems,
            ode_order,
            &x0.to_vec(),
            "",
            &ch_list,
            delta,
            Some(transient),
        )?
    };

    let data = data.ok_or_else(|| anyhow::anyhow!("Failed to generate data"))?;

    // Return data and parameters
    let params = DataGenerationParams {
        n_systems,
        a_values: vec![],
        b_values: vec![],
        c: 0.0,
        dt,
        length,
        transient,
        delta,
        dimension: dim,
        order: ode_order,
        initial_conditions: x0,
    };

    Ok((data, params))
}

/// Generate test data exactly matching the Julia implementation.
pub fn generate_test_data_matching_julia() -> Result<Array2<f64>> {
    // Exact parameters from run_first_DDA.jl
    let nr_syst = 4;

    let ros = Array2::from_shape_vec(
        (7, 3),
        vec![
            0, 0, 2,
            0, 0, 3,
            1, 0, 1,
            1, 0, 2,
            2, 0, 0,
            2, 0, 3,
            2, 1, 3,
        ],
    )?;

    // Model parameters
    let a123 = 0.21;
    let a456 = 0.20;
    let b1 = 0.2150;
    let b2 = 0.2020;
    let b4 = 0.4050;
    let b5 = 0.3991;
    let c = 5.7;

    let mod_par = vec![
        -1.0, -1.0, 1.0, a123, b1, -c, 1.0,
        -1.0, -1.0, 1.0, a123, b2, -c, 1.0,
        -1.0, -1.0, 1.0, a456, b4, -c, 1.0,
        -1.0, -1.0, 1.0, a456, b5, -c, 1.0,
    ];

    // Create model encoding
    let (mod_nr, dim, ode_order, _p) = create_mod_nr(&ros, nr_syst);

    // Integration parameters
    let trans = 20000;
    let dt = 0.05;
    let l = 20000;
    let delta = 2;

    // Set random seed and generate initial conditions
    let mut rng = StdRng::seed_from_u64(42);
    let x0: Vec<f64> = (0..dim * nr_syst).map(|_| rng.gen()).collect();

    // Channel list (x-coordinates only: 1, 4, 7, 10)
    let ch_list: Vec<usize> = (1..=dim * nr_syst).step_by(dim).collect();

    // Output filename
    let fn_data = "ROS_4.ascii";

    // Generate data
    integrate_ode_general(
        &mod_nr.to_vec().iter().map(|&x| x as i32).collect::<Vec<_>>(),
        &mod_par,
        dt,
        l,
        dim * nr_syst,
        ode_order,
        &x0,
        fn_data,
        &ch_list,
        delta,
        Some(trans),
    )?;

    // Load and return data
    let contents = fs::read_to_string(fn_data)?;
    let mut result = Vec::new();
    for line in contents.lines() {
        let row: Vec<f64> = line
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        result.push(row);
    }
    
    let rows = result.len();
    let cols = if rows > 0 { result[0].len() } else { 0 };
    let flat: Vec<f64> = result.into_iter().flatten().collect();
    
    Ok(Array2::from_shape_vec((rows, cols), flat)?)
}

/// Run external ODE integration using command line executable.
pub fn run_external_integration(
    mod_nr: &[i32],
    mod_par: &[f64],
    dt: f64,
    l: usize,
    dim: usize,
    order: usize,
    x0: &[f64],
    output_file: &str,
    ch_list: &[usize],
    delta: usize,
    trans: usize,
) -> Result<String> {
    // Platform-specific executable - check current dir first, then parent
    let executable = if cfg!(windows) {
        let exe_path = std::path::Path::new("i_ODE_general_BIG.exe");
        let parent_exe = std::path::Path::new("../i_ODE_general_BIG");
        
        if exe_path.exists() {
            exe_path
        } else if parent_exe.exists() {
            if !exe_path.exists() {
                std::fs::copy(parent_exe, &exe_path)?;
            }
            exe_path
        } else {
            return Err(anyhow::anyhow!("i_ODE_general_BIG executable not found"));
        }
    } else {
        let local_exe = std::path::Path::new("i_ODE_general_BIG");
        let parent_exe = std::path::Path::new("../i_ODE_general_BIG");
        
        if local_exe.exists() {
            local_exe
        } else if parent_exe.exists() {
            parent_exe
        } else {
            return Err(anyhow::anyhow!("i_ODE_general_BIG executable not found"));
        }
    };

    // Build command
    let mut cmd = Command::new(executable);
    
    cmd.arg("-MODEL");
    for m in mod_nr {
        cmd.arg(m.to_string());
    }
    
    cmd.arg("-PAR");
    for p in mod_par {
        cmd.arg(p.to_string());
    }
    
    cmd.arg("-ANF");
    for ic in x0 {
        cmd.arg(ic.to_string());
    }
    
    cmd.args(&["-dt", &dt.to_string()]);
    cmd.args(&["-L", &l.to_string()]);
    cmd.args(&["-DIM", &dim.to_string()]);
    cmd.args(&["-order", &order.to_string()]);
    
    if trans > 0 {
        cmd.args(&["-TRANS", &trans.to_string()]);
    }
    
    cmd.args(&["-FILE", output_file]);
    cmd.args(&["-DELTA", &delta.to_string()]);
    
    cmd.arg("-CH_list");
    for ch in ch_list {
        cmd.arg(ch.to_string());
    }

    // Execute command
    cmd.output()?;
    
    Ok(format!("{:?}", cmd))
}