# polars-random

Polars plugin for generating random distributions.

## Description

`polars-random` is a Rust plugin for the Polars DataFrame library that provides functionality to generate random numbers through a new dataframe namespace called "random". It supports generating random numbers from various distributions such as uniform, normal, and binomial.

You can set seeds, and pass the parameters as polars expressions or column names (as strings).

## Installation

To use `polars-random`, install it using your favourite tool:

```sh
uv add polars-random
```

```sh
poetry add polars-random
```
```sh
pip install polars-random
```


## Usage

Here are some examples of how to use the `polars-random` plugin:

### Uniform Distribution

```python
import polars as pl
import polars_random

df: pl.DataFrame = ...

random_series = (
    df
    .random.rand(low=1_000., high=2_000., name="rand")
    .random.rand(seed=42, name="rand_seed")
    .random.rand(
        low=pl.col("custom_low"),
        high=pl.col("custom_high"),
        name="rand_expr",
    )
    .random.rand(
        mean="custom_low",
        std="custom_high",
        name="rand_str",
    )
)
```

### Normal Distribution

```python
import polars as pl
import polars_random

df: pl.DataFrame = ...

random_series = (
    df
    .random.normal(mean=3., std=2., name="normal")
    .random.normal(seed=42, name="normal_seed")
    .random.normal(
        mean=pl.col("custom_mean"),
        std=pl.col("custom_std"),
        name="normal_expr",
    )
    .random.normal(
        mean="custom_mean",
        std="custom_std",
        name="normal_str",
    )
)
```

### Binomial Distribution

```python
import polars as pl
import polars_random

df: pl.DataFrame = ...

random_series = (
    df
    # Mandatory parameters n and p
    .random.binomial(n=100, p=.5, seed=42, name="binomial")
    .random.binomial(
        n=pl.col("custom_n"),
        p=pl.col("custom_p"),
        name="binomial_expr",
    )
    .random.binomial(
        n="n",
        p="p",
        name="binomial_str",
    )
)
```