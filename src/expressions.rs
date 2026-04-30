#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::prelude::*;
use rand_distr::{Binomial, Distribution, Normal, Uniform};
use serde::Deserialize;

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

#[derive(Deserialize)]
struct RandIntArgs {
    low: i64,
    high: i64,
    seed: Option<u64>,
}

#[derive(Deserialize)]
struct SeedArgs {
    seed: Option<u64>,
}

fn make_rng(seed: Option<u64>) -> SmallRng {
    match seed {
        Some(i) => SmallRng::seed_from_u64(i),
        None => SmallRng::from_os_rng(),
    }
}

#[polars_expr(output_type=Float64)]
fn rand(inputs: &[Series], kwargs: RandArgs) -> PolarsResult<Series> {
    let ca = inputs[0].f64()?;
    let count = ca.len();
    let mut rng = make_rng(kwargs.seed);
    let uniform: Uniform<f64> = Uniform::new(kwargs.low.unwrap_or(0.0), kwargs.high.unwrap_or(1.0))
        .map_err(|e| PolarsError::ComputeError(format!("invalid uniform range: {e}").into()))?;

    let mut out: Vec<f64> = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(uniform.sample(&mut rng));
    }
    Ok(Float64Chunked::from_vec(ca.name().clone(), out).into_series())
}

#[polars_expr(output_type=Float64)]
fn rand_expr(inputs: &[Series], kwargs: SeedArgs) -> PolarsResult<Series> {
    let low = inputs[0].f64()?;
    let high = inputs[1].f64()?;
    let mut rng = make_rng(kwargs.seed);

    let out = low
        .into_iter()
        .zip(high.into_iter())
        .map(|(l, h)| match (l, h) {
            (Some(l), Some(h)) => Uniform::new(l, h).ok().map(|u| u.sample(&mut rng)),
            _ => None,
        });
    Ok(Float64Chunked::from_iter_options(low.name().clone(), out).into_series())
}

#[polars_expr(output_type=Float64)]
fn normal(inputs: &[Series], kwargs: NormalArgs) -> PolarsResult<Series> {
    let ca = inputs[0].f64()?;
    let count = ca.len();
    let mut rng = make_rng(kwargs.seed);
    let mean = kwargs.mean.unwrap_or(0.0);
    let std = kwargs.std.unwrap_or(1.0);
    let normal: Normal<f64> = Normal::new(mean, std)
        .map_err(|e| PolarsError::ComputeError(format!("invalid normal params: {e}").into()))?;

    let mut out: Vec<f64> = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(normal.sample(&mut rng));
    }
    Ok(Float64Chunked::from_vec(ca.name().clone(), out).into_series())
}

#[polars_expr(output_type=Float64)]
fn normal_expr(inputs: &[Series], kwargs: SeedArgs) -> PolarsResult<Series> {
    let mean = inputs[0].f64()?;
    let std = inputs[1].f64()?;
    let mut rng = make_rng(kwargs.seed);

    let out = mean
        .into_iter()
        .zip(std.into_iter())
        .map(|(m, s)| match (m, s) {
            (Some(m), Some(s)) => Normal::new(m, s).ok().map(|n| n.sample(&mut rng)),
            _ => None,
        });
    Ok(Float64Chunked::from_iter_options(mean.name().clone(), out).into_series())
}

#[polars_expr(output_type=UInt64)]
fn binomial(inputs: &[Series], kwargs: BinomialArgs) -> PolarsResult<Series> {
    let ca = inputs[0].f64()?;
    let count = ca.len();
    let mut rng = make_rng(kwargs.seed);
    let binomial: Binomial = Binomial::new(kwargs.n, kwargs.p)
        .map_err(|e| PolarsError::ComputeError(format!("invalid binomial params: {e}").into()))?;

    let mut out: Vec<u64> = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(binomial.sample(&mut rng));
    }
    Ok(UInt64Chunked::from_vec(ca.name().clone(), out).into_series())
}

#[polars_expr(output_type=UInt64)]
fn binomial_expr(inputs: &[Series], kwargs: SeedArgs) -> PolarsResult<Series> {
    let n = inputs[0].u64()?;
    let p = inputs[1].f64()?;
    let mut rng = make_rng(kwargs.seed);

    let out = n
        .into_iter()
        .zip(p.into_iter())
        .map(|(n_, p_)| match (n_, p_) {
            (Some(n_), Some(p_)) => Binomial::new(n_, p_).ok().map(|b| b.sample(&mut rng)),
            _ => None,
        });
    Ok(UInt64Chunked::from_iter_options(n.name().clone(), out).into_series())
}

#[polars_expr(output_type=Int64)]
fn randint(inputs: &[Series], kwargs: RandIntArgs) -> PolarsResult<Series> {
    let ca = inputs[0].f64()?;
    let count = ca.len();
    let mut rng = make_rng(kwargs.seed);
    let uniform: Uniform<i64> = Uniform::new(kwargs.low, kwargs.high)
        .map_err(|e| PolarsError::ComputeError(format!("invalid randint range: {e}").into()))?;

    let mut out: Vec<i64> = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(uniform.sample(&mut rng));
    }
    Ok(Int64Chunked::from_vec(ca.name().clone(), out).into_series())
}

#[polars_expr(output_type=Int64)]
fn randint_expr(inputs: &[Series], kwargs: SeedArgs) -> PolarsResult<Series> {
    let low = inputs[0].i64()?;
    let high = inputs[1].i64()?;
    let mut rng = make_rng(kwargs.seed);

    let out = low
        .into_iter()
        .zip(high.into_iter())
        .map(|(l, h)| match (l, h) {
            (Some(l), Some(h)) => Uniform::new(l, h).ok().map(|u| u.sample(&mut rng)),
            _ => None,
        });
    Ok(Int64Chunked::from_iter_options(low.name().clone(), out).into_series())
}
