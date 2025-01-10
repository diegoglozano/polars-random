#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::prelude::*;
use std::f64::consts::PI;


#[polars_expr(output_type=Float64)]
fn rand(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].f64()?;
    let out = ca
        .apply(|opt_v: Option<f64>| opt_v.map(|_v: f64| {
            let mut rng = rand::thread_rng();
            rng.gen::<f64>()
        }
    ));
    Ok(out.into_series())
}


#[polars_expr(output_type=Float64)]
fn normal(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].f64()?;
    let out = ca
        .apply(|opt_v: Option<f64>| opt_v.map(|_v: f64| {
            let mut rng = rand::thread_rng();
            let mean: f64 = 0.0;
            let std_dev: f64 = 1.0;
            let variance: f64 = std_dev.powi(2);
            let uniform = rng.gen::<f64>();
            let exponent = -((uniform - mean).powi(2)) / (2.0 * variance);
            let denominator = std_dev * (2.0 * PI).sqrt();
            (1.0 / denominator) * exponent.exp()
        }
    ));
    Ok(out.into_series())
}
