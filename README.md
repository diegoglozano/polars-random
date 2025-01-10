# polars-random

Polars plugin for generating random distributions.

## Description

`polars-random` is a Rust plugin for the Polars DataFrame library that provides functionality to generate random numbers through a new expression namespace called "random". It supports generating random numbers from various distributions such as uniform, normal, and binomial.

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

df = pl.DataFrame({
    "values": [1.0, 2.0, 3.0]
})

random_series = (
    df
    .select([
        pl
        .col("values")
        .random.rand(seed=42)
    ])
)

print(random_series)
```

### Normal Distribution

```python
import polars as pl
import polars_random

df = pl.DataFrame({
    "values": [1.0, 2.0, 3.0]
})

random_series = (
    df
    .select([
        pl
        .col("values")
        .random.normal(mean=0.0, std=1.0, seed=42)
    ])
)

print(random_series)
```
