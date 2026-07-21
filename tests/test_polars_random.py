import polars as pl
import pytest

import polars_random as pr  # noqa: F401  (also registers namespaces)


@pytest.fixture(autouse=True)
def _reset_global_seed():
    """Keep the module-global seed from leaking between tests."""
    pr._GLOBAL_RNG = None
    yield
    pr._GLOBAL_RNG = None


# ---------------- DataFrame namespace (existing) ----------------


def test_df_normal_mean():
    df = pl.DataFrame({"a": range(1_000_000)}).random.normal(mean=0.0, std=1.0)
    assert abs(df.select(pl.col("normal").mean()).item() - 0.0) < 0.01


def test_df_uniform_mean():
    df = pl.DataFrame({"a": range(1_000_000)}).random.rand(low=0.0, high=1.0)
    assert abs(df.select(pl.col("rand").mean()).item() - 0.5) < 0.01


def test_df_binomial_mean():
    df = pl.DataFrame({"a": range(1_000_000)}).random.binomial(n=10, p=0.5)
    assert abs(df.select(pl.col("binomial").mean()).item() - 5) < 0.01


def test_df_randint_range():
    df = pl.DataFrame({"a": range(10_000)}).random.randint(low=0, high=10, seed=42)
    out = df["randint"]
    assert out.dtype == pl.Int64
    assert out.min() >= 0
    assert out.max() < 10


def test_df_custom_name():
    df = pl.DataFrame({"a": [1]}).random.rand(seed=0, name="foo")
    assert "foo" in df.columns


def test_df_uniform_alias():
    df = pl.DataFrame({"a": range(100)})
    a = df.random.rand(seed=42)["rand"].to_list()
    b = df.random.uniform(seed=42)["rand"].to_list()
    assert a == b


# ---------------- Top-level helpers (new) ----------------


def test_top_level_returns_series_with_size():
    s = pr.rand(size=10, seed=42)
    assert isinstance(s, pl.Series)
    assert s.len() == 10
    assert s.dtype == pl.Float64


def test_top_level_returns_expr_without_size():
    e = pr.rand(seed=42)
    assert isinstance(e, pl.Expr)
    df = pl.DataFrame({"x": range(5)}).with_columns(r=e)
    assert df["r"].len() == 5


def test_top_level_normal_size():
    s = pr.normal(mean=10.0, std=0.0001, size=100, seed=42)
    assert abs(s.mean() - 10.0) < 0.001


def test_top_level_randint_size():
    s = pr.randint(low=-5, high=5, size=10_000, seed=42)
    assert s.dtype == pl.Int64
    assert s.min() >= -5 and s.max() < 5


def test_top_level_binomial_size():
    s = pr.binomial(n=4, p=1.0, size=10, seed=42)
    assert (s == 4).all()


def test_top_level_seed_reproducible():
    a = pr.rand(size=20, seed=123)
    b = pr.rand(size=20, seed=123)
    assert a.to_list() == b.to_list()


def test_top_level_zero_size():
    assert pr.rand(size=0, seed=0).len() == 0


def test_top_level_negative_size_raises():
    with pytest.raises(ValueError, match="non-negative"):
        pr.rand(size=-1)


# ---------------- Expression namespace (new) ----------------


def test_expr_namespace_normal():
    df = pl.DataFrame({"x": range(1_000_000)})
    out = df.with_columns(noise=pl.col("x").random.normal(mean=0.0, std=1.0))
    assert abs(out["noise"].mean() - 0.0) < 0.01


def test_expr_namespace_randint():
    df = pl.DataFrame({"x": range(1000)})
    out = df.with_columns(r=pl.col("x").random.randint(low=10, high=20, seed=1))
    assert out["r"].min() >= 10 and out["r"].max() < 20


def test_expr_namespace_uniform_alias():
    df = pl.DataFrame({"x": range(50)})
    a = df.with_columns(r=pl.col("x").random.rand(seed=7))["r"].to_list()
    b = df.with_columns(r=pl.col("x").random.uniform(seed=7))["r"].to_list()
    assert a == b


# ---------------- LazyFrame namespace (new) ----------------


def test_lazyframe_namespace():
    lf = pl.DataFrame({"x": range(100)}).lazy()
    out = lf.random.rand(seed=42, name="r").collect()
    assert "r" in out.columns
    assert out.height == 100


def test_lazyframe_chained():
    lf = (
        pl.DataFrame({"x": range(50)})
        .lazy()
        .random.rand(seed=1, name="u")
        .random.normal(seed=2, name="n")
        .random.binomial(n=5, p=0.5, seed=3, name="b")
        .random.randint(low=0, high=10, seed=4, name="i")
    )
    out = lf.collect()
    assert {"u", "n", "b", "i"}.issubset(out.columns)


# ---------------- Null safety (new) ----------------


def test_rand_null_inputs_propagate():
    df = pl.DataFrame({"lo": [0.0, None, 0.0], "hi": [1.0, 1.0, None]})
    out = df.random.rand(low="lo", high="hi", seed=0)
    nulls = out["rand"].is_null().to_list()
    assert nulls == [False, True, True]


def test_normal_null_inputs_propagate():
    df = pl.DataFrame({"m": [0.0, None], "s": [1.0, 1.0]})
    out = df.random.normal(mean="m", std="s", seed=0)
    assert out["normal"].is_null().to_list() == [False, True]


def test_binomial_null_inputs_propagate():
    df = pl.DataFrame({"n": [10, None], "p": [0.5, 0.5]})
    out = df.random.binomial(n="n", p="p", seed=0)
    assert out["binomial"].is_null().to_list() == [False, True]


def test_randint_null_inputs_propagate():
    df = pl.DataFrame({"lo": [0, None], "hi": [10, 10]})
    out = df.random.randint(low="lo", high="hi", seed=0)
    assert out["randint"].is_null().to_list() == [False, True]


# ---------------- Validation paths ----------------


def test_negative_seed_raises():
    with pytest.raises(ValueError, match="non-negative"):
        pl.DataFrame({"a": [1]}).random.rand(seed=-1)


def test_invalid_probability_raises():
    with pytest.raises(ValueError, match="between 0 and 1"):
        pl.DataFrame({"a": [1]}).random.binomial(n=10, p=1.5)


def test_mixed_scalar_and_column_raises():
    df = pl.DataFrame({"a": [1.0]})
    with pytest.raises(ValueError, match="must be either"):
        df.random.rand(low=0.0, high="a")


# ---------------- Per-row parameters ----------------


def test_per_row_uniform():
    df = pl.DataFrame({"lo": [0.0] * 10, "hi": [1.0, 2.0] * 5})
    out = df.random.rand(low="lo", high="hi", seed=42)
    assert out["rand"].len() == 10


def test_per_row_normal_via_expr():
    df = pl.DataFrame({"m": [0.0] * 10, "s": [1.0] * 10})
    out = df.with_columns(n=pl.col("m").random.normal(mean=pl.col("m"), std=pl.col("s"), seed=42))
    assert out["n"].len() == 10


# ---------------- Global seed (new) ----------------


def test_global_seed_makes_seedless_reproducible():
    df = pl.DataFrame({"x": range(100)})
    pr.set_random_seed(42)
    a = df.with_columns(r=pr.normal()).get_column("r").to_list()
    pr.set_random_seed(42)
    b = df.with_columns(r=pr.normal()).get_column("r").to_list()
    assert a == b


def test_global_seed_distinct_expressions_are_independent():
    # Two seedless draws in the same query consume the global generator
    # separately, so they must not be byte-for-byte identical.
    df = pl.DataFrame({"x": range(1_000)})
    pr.set_random_seed(42)
    out = df.with_columns(a=pr.normal(), b=pr.normal())
    assert out["a"].to_list() != out["b"].to_list()


def test_explicit_seed_overrides_global():
    df = pl.DataFrame({"x": range(100)})
    pr.set_random_seed(42)
    with_global = df.with_columns(r=pr.rand()).get_column("r").to_list()
    explicit = df.with_columns(r=pr.rand(seed=123)).get_column("r").to_list()
    reference = df.with_columns(r=pr.rand(seed=123)).get_column("r").to_list()
    assert explicit == reference
    assert explicit != with_global


def test_global_seed_applies_across_entry_points():
    # Top-level size=, expr namespace, df namespace and lazy all honor it.
    pr.set_random_seed(7)
    top = pr.rand(size=50).to_list()
    pr.set_random_seed(7)
    expr = (
        pl.DataFrame({"x": range(50)})
        .with_columns(r=pl.col("x").random.rand())
        .get_column("r")
        .to_list()
    )
    pr.set_random_seed(7)
    lazy = (
        pl.DataFrame({"x": range(50)})
        .lazy()
        .random.rand(name="r")
        .collect()
        .get_column("r")
        .to_list()
    )
    assert top == expr == lazy


def test_set_random_seed_negative_raises():
    with pytest.raises(ValueError, match="non-negative"):
        pr.set_random_seed(-1)


def test_no_global_seed_leaves_entropy_default():
    # Without a global seed, seedless draws stay entropy-based (independent).
    df = pl.DataFrame({"x": range(1_000)})
    a = df.with_columns(r=pr.rand()).get_column("r").to_list()
    b = df.with_columns(r=pr.rand()).get_column("r").to_list()
    assert a != b
