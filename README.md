# polars-random

[![PyPI version](https://img.shields.io/pypi/v/polars-random.svg)](https://pypi.org/project/polars-random/)
[![Python versions](https://img.shields.io/pypi/pyversions/polars-random.svg)](https://pypi.org/project/polars-random/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/diegoglozano/polars-random/blob/main/LICENSE)
[![CI](https://github.com/diegoglozano/polars-random/actions/workflows/testing.yml/badge.svg)](https://github.com/diegoglozano/polars-random/actions/workflows/testing.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://diegoglozano.github.io/polars-random/)

**Generate random numbers and statistical distributions natively in [Polars](https://pola.rs/) DataFrames** — a NumPy-style random API exposed as first-class Polars expressions, with reproducible seeds and per-row parameters.

`polars-random` is a Rust plugin that registers a `.random` namespace on `pl.DataFrame`. Use it to add columns of uniform, normal, or binomial draws — with parameters that can be Python literals, column names, or arbitrary Polars expressions.

```python
import polars as pl
import polars_random  # registers df.random

df = pl.DataFrame({"id": range(5)})

df.random.normal(mean=0.0, std=1.0, seed=42, name="noise")
# shape: (5, 2)
# ┌─────┬───────────┐
# │ id  ┆ noise     │
# │ --- ┆ ---       │
# │ i64 ┆ f64       │
# ╞═════╪═══════════╡
# │ 0   ┆  0.49671… │
# │ 1   ┆ -0.13826… │
# │ 2   ┆  0.64769… │
# │ 3   ┆  1.52303… │
# │ 4   ┆ -0.23415… │
# └─────┴───────────┘
```

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

| NumPy                                    | polars-random                                            |
| ---------------------------------------- | -------------------------------------------------------- |
| `np.random.uniform(low, high, size=n)`   | `df.random.rand(low=low, high=high)`                     |
| `np.random.normal(mean, std, size=n)`    | `df.random.normal(mean=mean, std=std)`                   |
| `np.random.binomial(n, p, size=size)`    | `df.random.binomial(n=n, p=p)`                           |
| `np.random.seed(42)` (global)            | `seed=42` per call                                       |
| Different params per row (loop / vectorize manually) | Pass a column name or `pl.col(...)` as the parameter |

The output length always matches the DataFrame's height — no `size=` argument needed.

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

## Documentation

Full API reference: <https://diegoglozano.github.io/polars-random/>

## License

[MIT](LICENSE)
