import polars as pl
import pytest

import polars_random  # noqa


@pytest.mark.parametrize(
    ("func_name", "params", "target"),
    [
        ("normal", dict(mean=0.0, std=1.0), 0.0),
        ("rand", dict(low=0.0, high=1.0), 0.5),
        ("binomial", dict(n=10, p=0.5), 5.0),
    ],
)
@pytest.mark.parametrize(
    ("lazy", "streaming"),
    [
        (False, False),
    ],
)
def test_all(
    lazy: bool, streaming: bool, func_name: str, params: dict[str, float], target: float
) -> None:
    # Initialize the dataframe
    df: pl.DataFrame | pl.LazyFrame = pl.DataFrame(
        {
            "a": range(1_000_000),
        }
    )
    if lazy:
        df = df.lazy()

    # Add the random column
    df = getattr(df.random, func_name)(**params, name="rand")  # type: ignore

    # Calculate the mean
    df_mean = df.select(pl.col("rand").mean())
    if lazy:
        df_mean = df_mean.collect(streaming=streaming)
    mean = df_mean.item()

    assert abs(mean - target) < 0.01
