# polars-random docs

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

For every available distribution, parameters can be passed as `polars expressions` or native python objects (`int`, `float`, `string`...).

Here are some examples of how to use the `polars-random` plugin for generating uniform distributions:

```python
import polars as pl
# This will automatically register .random
# in pl.DataFrame namespace
import polars_random

df: pl.DataFrame = ...
```

If we want to generate a new uniform column called `rand` based on some parameters:
```python
(
    df
    .random.rand(
        low=1_000.,
        high=2_000.,
        name="rand",
    )
)
```

We can also add a `seed` and make the generation reproducible. In the following case, we are using default parameters (`low=1.` and `high=1.`) and generating a uniform distribution called `rand_seed`:
```python
(
    df
    .random.rand(
        seed=42, 
        name="rand_seed",
    )
```

If we want custom parameters, we can use `polars expressions`. Let's say we have two columns called `custom_low` and `custom_high`. For generating a new column, we can use either the expression `pl.col("custom_low")` or a python string `"custom_low"`:
```python
(
    df
    .random.rand(
        low=pl.col("custom_low"),
        high=pl.col("custom_high"),
        name="rand_expr",
    )
    .random.rand(
        low="custom_low",
        high="custom_high",
        name="rand_str",
    )
)
```

## Distributions

### Uniform distribution

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
