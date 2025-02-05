import polars as pl

import polars_random  # noqa


def test_normal():
    df = pl.DataFrame(
        {
            "a": range(1_000_000),
        }
    ).random.normal(mean=0.0, std=1.0)  # type: ignore
    assert abs(df.select(pl.col("normal").mean()).item() - 0.0) < 0.01


def test_uniform():
    df = pl.DataFrame(
        {
            "a": range(1_000_000),
        }
    ).random.rand(low=0.0, high=1.0)  # type: ignore
    assert abs(df.select(pl.col("rand").mean()).item() - 0.5) < 0.01


def test_binomial():
    df = pl.DataFrame(
        {
            "a": range(1_000_000),
        }
    ).random.binomial(n=10, p=0.5)  # type: ignore
    assert abs(df.select(pl.col("binomial").mean()).item() - 5) < 0.01
