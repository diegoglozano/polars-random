"""Random sequences must not restart when Polars partitions the input."""

import polars as pl
import pytest
from polars.testing import assert_series_equal

import polars_random as pr


@pytest.fixture(autouse=True)
def reset_global_seed():
    pr._GLOBAL_RNG = None
    yield
    pr._GLOBAL_RNG = None


@pytest.fixture(params=["rand", "normal", "randint", "binomial"])
def distribution(request):
    name = request.param
    params = {
        "rand": {"low": 0.0, "high": 1.0},
        "normal": {"mean": 0.0, "std": 1.0},
        "randint": {"low": 0, "high": 2**60},
        "binomial": {"n": 100, "p": 0.5},
    }[name]
    return name, params


@pytest.mark.parametrize("column_params", [False, True], ids=["scalar", "column"])
def test_million_rows_match_across_chunks_and_engines(distribution, column_params):
    name, params = distribution
    size = 1_000_003  # Uneven partitions also exercise the final chunk.
    df = pl.DataFrame({key: pl.repeat(value, size, eager=True) for key, value in params.items()})
    df = pl.concat(df.iter_slices(83_333), rechunk=False)
    assert df.n_chunks() > 1
    kwargs = {key: pl.col(key) for key in params} if column_params else params
    expected = getattr(pr, name)(**params, size=size, seed=42).alias("draw")

    expr = getattr(pr, name)(**kwargs, seed=42).alias("draw")
    anchored = getattr(pl.col(df.columns[0]).random, name)(**kwargs, seed=42).alias("draw")
    actuals = [df.select(expr).to_series(), df.select(anchored).to_series()]
    query = getattr(df.lazy().random, name)(**kwargs, seed=42, name="draw").select("draw")
    actuals.extend(
        query.collect(engine=engine).to_series() for engine in ["in-memory", "streaming"]
    )
    # Collecting the same plan twice must rewind its explicit seed.
    actuals.append(query.collect(engine="streaming").to_series())

    for actual in actuals:
        assert_series_equal(actual, expected)
        if name != "binomial":
            # Binomial draws naturally have low cardinality. The other test
            # distributions have enough support for almost every draw to differ.
            assert actual.n_unique() > size * 0.99


def test_global_seed_does_not_restart_in_streaming():
    df = pl.DataFrame({"id": range(1_000_000)})

    def query():
        return df.lazy().with_columns(a=pr.rand(), b=pr.normal()).select("a", "b")

    pr.set_random_seed(42)
    expected = query().collect(engine="in-memory")
    pr.set_random_seed(42)
    actual = query().collect(engine="streaming")
    assert actual.equals(expected)
    assert actual["a"].n_unique() > 990_000
    assert actual["b"].n_unique() > 990_000


def test_varying_column_params_with_nulls(distribution):
    name, params = distribution
    size = 100_003
    # Nulls in either input and varying parameters exercise the RNG's conditional
    # consumption, including distributions that use rejection sampling.
    df = pl.DataFrame(
        {
            key: [None if i % (11 + offset) == 0 else value * (1 + i % 2) for i in range(size)]
            for offset, (key, value) in enumerate(params.items())
        }
    )
    expr = getattr(pr, name)(**{key: key for key in params}, seed=7).alias("draw")
    expected = df.select(expr).to_series()
    chunked = pl.concat(df.iter_slices(7_777), rechunk=False)
    actual = chunked.lazy().select(expr).collect(engine="streaming").to_series()
    assert_series_equal(actual, expected)
    assert actual.null_count() == df.select(pl.any_horizontal(pl.all().is_null()).sum()).item()


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_seedless_expressions_stay_independent(distribution, engine):
    name, params = distribution
    draw = getattr(pr, name)
    df = pl.DataFrame({"id": range(10_000)})
    # Identical entropy-based expressions must not be combined by Polars' common
    # subexpression elimination.
    query = df.lazy().select(a=draw(**params), b=draw(**params))
    first = query.collect(engine=engine)
    second = query.collect(engine=engine)
    assert isinstance(first, pl.DataFrame)
    assert isinstance(second, pl.DataFrame)
    assert not first["a"].equals(first["b"].alias("a"))
    assert not first.equals(second)
