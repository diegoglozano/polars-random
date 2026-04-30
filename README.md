# polars-random

[![PyPI version](https://img.shields.io/pypi/v/polars-random.svg)](https://pypi.org/project/polars-random/)
[![Python versions](https://img.shields.io/pypi/pyversions/polars-random.svg)](https://pypi.org/project/polars-random/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/diegoglozano/polars-random/blob/main/LICENSE)
[![CI](https://github.com/diegoglozano/polars-random/actions/workflows/testing.yml/badge.svg)](https://github.com/diegoglozano/polars-random/actions/workflows/testing.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://diegoglozano.github.io/polars-random/)

**Generate random numbers and statistical distributions natively in [Polars](https://pola.rs/) DataFrames** — a NumPy-style random API exposed as first-class Polars expressions, with reproducible seeds and per-row parameters.

`polars-random` is a Rust plugin offering four equivalent entry points so it composes naturally with the rest of polars:

| Use case                                           | API                                                                |
| -------------------------------------------------- | ------------------------------------------------------------------ |
| "Add a column of random draws to a DataFrame"      | `df.random.<dist>(...)`                                            |
| Same thing, lazy                                   | `lf.random.<dist>(...)`                                            |
| Inside any expression / `with_columns` / `select`  | `pl.col("x").random.<dist>(...)` &nbsp;or&nbsp; `polars_random.<dist>(...)` |
| Just give me N values as a Series                  | `polars_random.<dist>(..., size=N)`                                |

```python
import polars as pl
import polars_random as pr  # registers DataFrame/LazyFrame/Expr namespaces

# 1. eager Series
pr.normal(mean=0.0, std=1.0, size=5, seed=42)

# 2. as a polars expression in any context
df = pl.DataFrame({"id": range(5)})
df.with_columns(noise=pr.normal(mean=0.0, std=1.0, seed=42))
df.with_columns(noise=pl.col("id").random.normal(seed=42))

# 3. as a DataFrame method (returns a new DataFrame with the column appended)
df.random.normal(mean=0.0, std=1.0, seed=42, name="noise")

# 4. inside a lazy pipeline
df.lazy().random.normal(seed=42, name="noise").collect()
```

Available distributions: `rand` / `uniform`, `normal`, `binomial`, `randint`. Every parameter (`low`, `high`, `mean`, `std`, `n`, `p`) accepts a Python scalar, a column name (`"my_col"`), or any `pl.Expr`. Nulls in column-valued parameters propagate as null in the output (no panic).

## Why polars-random?

- **Polars-native** — outputs are regular Polars columns, composable with the rest of your pipeline (no NumPy round-trips).
- **Per-row parameters** — `mean`, `std`, `low`, `high`, `n`, `p` can come from other columns, so each row can be drawn from a different distribution.
- **Reproducible** — pass `seed=...` for deterministic draws.
- **Fast** — implemented in Rust on top of `rand` / `rand_distr`.

## Installation

```sh
uv add polars-random
```

```sh
poetry add polars-random
```

```sh
pip install polars-random
```

## How it works (mental model)

Every distribution follows the same shape:

```python
df.random.<distribution>(<params>, seed=None, name=None)
```

- `<params>` are the distribution's parameters (e.g. `low`/`high`, `mean`/`std`, `n`/`p`).
- Each parameter accepts **a Python literal**, **a column name as a string**, or **a Polars expression** (`pl.col(...)`, arithmetic, etc.). Within a single call, all distribution parameters must be the same kind — either all literals or all expressions/column-names (no mixing).
- `seed` makes the draw reproducible. Omit it for entropy-based randomness.
- `name` is the new column's name. Defaults to the distribution name (`"rand"`, `"normal"`, `"binomial"`).
- The result is a new `pl.DataFrame` with the column appended. Calls chain.

## Coming from NumPy?

| NumPy                                    | polars-random                                                       |
| ---------------------------------------- | ------------------------------------------------------------------- |
| `np.random.uniform(low, high, size=n)`   | `pr.rand(low=low, high=high, size=n)` &nbsp;or&nbsp; `df.random.rand(low=low, high=high)` |
| `np.random.normal(mean, std, size=n)`    | `pr.normal(mean=mean, std=std, size=n)`                             |
| `np.random.binomial(n, p, size=size)`    | `pr.binomial(n=n, p=p, size=size)`                                  |
| `np.random.randint(low, high, size=n)`   | `pr.randint(low=low, high=high, size=n)`                            |
| `np.random.seed(42)` (global)            | `seed=42` per call                                                  |
| Different params per row (loop / vectorize manually) | Pass a column name or `pl.col(...)` as the parameter      |

When used as a DataFrame/LazyFrame method or via the `pl.col(...).random` namespace, the output length is taken from the parent — no `size=` needed. Use `size=N` only with the top-level functions for "give me N values without a frame."

## Distributions

### `df.random.rand` (uniform) · also aliased as `df.random.uniform`

| Parameter | Type                                    | Default | Description                  |
| --------- | --------------------------------------- | ------- | ---------------------------- |
| `low`     | `float`, `str`, `pl.Expr`, or `None`    | `0.0`   | Lower bound (inclusive).     |
| `high`    | `float`, `str`, `pl.Expr`, or `None`    | `1.0`   | Upper bound (exclusive).     |
| `seed`    | `int` or `None`                         | `None`  | Reproducible draws.          |
| `name`    | `str` or `None`                         | `"rand"`| Output column name.          |

```python
import polars as pl
import polars_random

df = pl.DataFrame({
    "custom_low":  [0.0, 10.0, 100.0],
    "custom_high": [1.0, 20.0, 200.0],
})

(
    df
    # Scalar parameters
    .random.rand(low=1_000., high=2_000., seed=42, name="rand_scalar")
    # Default range [0, 1)
    .random.rand(seed=42, name="rand_default")
    # Per-row parameters via expression
    .random.rand(low=pl.col("custom_low"), high=pl.col("custom_high"), seed=42, name="rand_expr")
    # Per-row parameters via column name
    .random.rand(low="custom_low", high="custom_high", seed=42, name="rand_str")
)
```

### `df.random.normal`

| Parameter | Type                                    | Default    | Description                          |
| --------- | --------------------------------------- | ---------- | ------------------------------------ |
| `mean`    | `float`, `str`, `pl.Expr`, or `None`    | `0.0`      | Mean of the normal distribution.     |
| `std`     | `float`, `str`, `pl.Expr`, or `None`    | `1.0`      | Standard deviation (must be `> 0`).  |
| `seed`    | `int` or `None`                         | `None`     | Reproducible draws.                  |
| `name`    | `str` or `None`                         | `"normal"` | Output column name.                  |

```python
import polars as pl
import polars_random

df = pl.DataFrame({
    "custom_mean": [0.0, 5.0, -3.0],
    "custom_std":  [1.0, 2.0, 0.5],
})

(
    df
    .random.normal(mean=3., std=2., seed=42, name="normal_scalar")
    .random.normal(seed=42, name="normal_default")  # mean=0, std=1
    .random.normal(mean=pl.col("custom_mean"), std=pl.col("custom_std"), seed=42, name="normal_expr")
    .random.normal(mean="custom_mean", std="custom_std", seed=42, name="normal_str")
)
```

### `df.random.binomial`

| Parameter | Type                          | Default      | Description                                       |
| --------- | ----------------------------- | ------------ | ------------------------------------------------- |
| `n`       | `int`, `str`, or `pl.Expr`    | *(required)* | Number of trials.                                 |
| `p`       | `float`, `str`, or `pl.Expr`  | *(required)* | Probability of success on each trial (`0 ≤ p ≤ 1`). |
| `seed`    | `int` or `None`               | `None`       | Reproducible draws.                               |
| `name`    | `str` or `None`               | `"binomial"` | Output column name.                               |

```python
import polars as pl
import polars_random

df = pl.DataFrame({
    "n": [10, 50, 100],
    "p": [0.1, 0.5, 0.9],
})

(
    df
    .random.binomial(n=100, p=.5, seed=42, name="binomial_scalar")
    .random.binomial(n=pl.col("n"), p=pl.col("p"), seed=42, name="binomial_expr")
    .random.binomial(n="n", p="p", seed=42, name="binomial_str")
)
```

### `df.random.randint`

Uniform random integers in `[low, high)` (high is exclusive, matching `numpy.random.randint`).

| Parameter | Type                          | Default      | Description                  |
| --------- | ----------------------------- | ------------ | ---------------------------- |
| `low`     | `int`, `str`, or `pl.Expr`    | `0`          | Lower bound (inclusive).     |
| `high`    | `int`, `str`, or `pl.Expr`    | `2`          | Upper bound (exclusive).     |
| `seed`    | `int` or `None`               | `None`       | Reproducible draws.          |
| `name`    | `str` or `None`               | `"randint"`  | Output column name.          |

```python
df.random.randint(low=0, high=10, seed=42)            # one column, scalar bounds
df.random.randint(low="lo", high="hi", seed=42)       # per-row bounds via columns
```

## Beyond `df.random` — same kernel, four entry points

```python
import polars as pl
import polars_random as pr

# 1. Top-level: returns a Series of N random values (NumPy-style).
pr.normal(mean=0, std=1, size=1_000, seed=42)

# 2. Top-level inside any expression (length comes from the surrounding context).
df.with_columns(noise=pr.normal(mean=0, std=1, seed=42))

# 3. Expression namespace — anchor random draws to an existing column.
df.with_columns(noise=pl.col("id").random.normal(mean=0, std=1, seed=42))

# 4. LazyFrame — keep random draws inside a lazy plan.
df.lazy().random.binomial(n=10, p=0.5, seed=42, name="trials").collect()
```

When a parameter is column-valued (`pl.col(...)`, a column name, or any expression) and contains nulls, the output is null at those rows instead of raising.

## Documentation

Full API reference: <https://diegoglozano.github.io/polars-random/>

## License

[MIT](LICENSE)
