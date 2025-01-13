#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::prelude::*;
use serde::Deserialize;
use rand_distr::{Binomial, Distribution, Normal, Uniform};

#[derive(Deserialize)]
struct RandArgs {
    low: Option<f64>,
    high: Option<f64>,
    seed: Option<u64>,
}

#[derive(Deserialize)]
struct NormalArgs {
    mean: Option<f64>,
    std: Option<f64>,
    seed: Option<u64>,
}

#[derive(Deserialize)]
struct BinomialArgs {
    n: u64,
    p: f64,
    seed: Option<u64>,
}

#[polars_expr(output_type=Float64)]
fn rand(inputs: &[Series], kwargs: RandArgs) -> PolarsResult<Series> {
    let ca = inputs[0].f64()?;
    let count = ca.len();
    let mut out: Vec<f64> = Vec::with_capacity(count);
    
    let mut rng = match kwargs.seed {
        Some(i) => SmallRng::seed_from_u64(i),
        None => SmallRng::from_entropy(),
    };
    let uniform: Uniform<f64> = Uniform::new(
        kwargs.low.unwrap_or(0.0),
        kwargs.high.unwrap_or(1.0),
    );

    for _ in 0..count {
        out.push(uniform.sample(&mut rng));
    }
    Ok(Float64Chunked::from_vec(ca.name().clone(), out).into_series())
}


#[polars_expr(output_type=Float64)]
fn normal(inputs: &[Series], kwargs: NormalArgs) -> PolarsResult<Series> {
    let ca = inputs[0].f64()?;
    let count = ca.len();
    let mut out: Vec<f64> = Vec::with_capacity(count);
    
    let mut rng = match kwargs.seed {
        Some(i) => SmallRng::seed_from_u64(i),
        None => SmallRng::from_entropy(),
    };
    let mean = kwargs.mean.unwrap_or(0.0);
    let std = kwargs.std.unwrap_or(1.0);
    let normal: Normal<f64> = Normal::new(mean, std).unwrap();

    for _ in 0..count {
        out.push(normal.sample(&mut rng));
    }
    Ok(Float64Chunked::from_vec(ca.name().clone(), out).into_series())
}


#[polars_expr(output_type=UInt64)]
fn binomial(inputs: &[Series], kwargs: BinomialArgs) -> PolarsResult<Series> {
    let ca = inputs[0].f64()?;
    let count = ca.len();
    let mut out: Vec<u64> = Vec::with_capacity(count);
    
    let mut rng = match kwargs.seed {
        Some(i) => SmallRng::seed_from_u64(i),
        None => SmallRng::from_entropy(),
    };
    let n = kwargs.n;
    let p = kwargs.p;
    let binomial: Binomial = Binomial::new(n, p).unwrap();

    for _ in 0..count {
        out.push(binomial.sample(&mut rng));
    }
    Ok(UInt64Chunked::from_vec(ca.name().clone(), out).into_series())
}
