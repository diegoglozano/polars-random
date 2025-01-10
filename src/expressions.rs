#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::prelude::*;
use serde::Deserialize;
use rand_distr::{Normal, Distribution};

#[derive(Deserialize)]
struct RandArgs {
    seed: Option<u64>,
}

#[derive(Deserialize)]
struct NormalArgs {
    mean: Option<f64>,
    std: Option<f64>,
    seed: Option<u64>,
}

#[polars_expr(output_type=Float64)]
fn rand(inputs: &[Series], kwargs: RandArgs) -> PolarsResult<Series> {
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
fn normal(inputs: &[Series], kwargs: NormalArgs) -> PolarsResult<Series> {
    let ca = inputs[0].f64()?;
    let out = ca
        .apply(|opt_v: Option<f64>| opt_v.map(|_v: f64| {
            let mean = kwargs.mean.unwrap_or(0.0);
            let std = kwargs.std.unwrap_or(1.0);
            let mut rng = rand::thread_rng();
            let normal: Normal<f64> = Normal::new(mean, std).unwrap();
            normal.sample(&mut rng)
        }
    ));
    Ok(out.into_series())
}
