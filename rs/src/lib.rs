#[macro_use]
extern crate ndarray;

pub mod dda_functions;
pub mod dda_st;
pub mod dda_ct;
pub mod dda_de;
pub mod generate_data;

// Re-export commonly used items
pub use dda_functions::{
    deriv_all,
    ensure_directory_exists,
    integrate_ode_general,
    generate_monomial_indices,
    generate_monomial_list,
    create_model,
    create_mod_nr,
    create_coupling_mod_nr,
    add_noise,
};

pub use dda_st::{
    compute_st_single,
    compute_st_multiple,
    run_dda_st_external,
};

pub use dda_ct::{
    compute_ct_pair,
    compute_ct_multiple,
    run_dda_ct_external,
};

pub use dda_de::{
    compute_dynamical_ergodicity,
    analyze_ergodicity_statistics,
    compare_with_external_de,
    run_full_de_analysis,
    ErgodicityStat,
};

pub use generate_data::{
    generate_roessler_data,
    generate_custom_system_data,
    generate_test_data_matching_julia,
    run_external_integration,
    DataGenerationParams,
};