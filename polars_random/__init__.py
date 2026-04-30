from __future__ import annotations

from pathlib import Path
from typing import Union

import polars as pl
from polars.plugins import register_plugin_function

from polars_random._internal import __version__ as __version__

__all__ = [
    "binomial",
    "normal",
    "rand",
    "randint",
    "uniform",
]

LIB = Path(__file__).parent

FloatParam = Union[float, int, pl.Expr, str, None]
IntParam = Union[int, pl.Expr, str, None]


def _check_seed(seed: int | None) -> None:
    if seed is not None and seed < 0:
        raise ValueError("Seed must be a non-negative integer")


def _check_probability(prob: float) -> None:
    if prob < 0 or prob > 1:
        raise ValueError("Probability must be between 0 and 1")


def _is_columnar(value: object) -> bool:
    return isinstance(value, (pl.Expr, str))


def _to_expr(value: FloatParam | IntParam, dtype: pl.DataType | type[pl.DataType]) -> pl.Expr:
    if isinstance(value, str):
        return pl.col(value).cast(dtype)
    if isinstance(value, pl.Expr):
        return value.cast(dtype)
    raise TypeError(f"expected column name or expression, got {type(value).__name__}")


def _length_anchor(over: pl.Expr | None) -> pl.Expr:
    if over is None:
        return pl.int_range(pl.len()).cast(pl.Float64)
    return over.cast(pl.Float64)


def _consistent_pair(a: object, b: object, names: tuple[str, str]) -> None:
    if _is_columnar(a) != _is_columnar(b):
        raise ValueError(
            f"Both {names[0]} and {names[1]} must be either expressions/column names "
            f"or scalars (a mix is not allowed)."
        )


def _rand_expr(
    low: FloatParam = None,
    high: FloatParam = None,
    seed: int | None = None,
    over: pl.Expr | None = None,
) -> pl.Expr:
    _check_seed(seed)
    _consistent_pair(low, high, ("low", "high"))

    if _is_columnar(low):
        return register_plugin_function(
            args=[_to_expr(low, pl.Float64), _to_expr(high, pl.Float64)],
            plugin_path=LIB,
            function_name="rand_expr",
            is_elementwise=True,
            kwargs={"seed": seed},
        )
    return register_plugin_function(
        args=[_length_anchor(over)],
        plugin_path=LIB,
        function_name="rand",
        is_elementwise=True,
        kwargs={"low": low, "high": high, "seed": seed},
    )


def _normal_expr(
    mean: FloatParam = 0.0,
    std: FloatParam = 1.0,
    seed: int | None = None,
    over: pl.Expr | None = None,
) -> pl.Expr:
    _check_seed(seed)
    _consistent_pair(mean, std, ("mean", "std"))

    if _is_columnar(mean):
        return register_plugin_function(
            args=[_to_expr(mean, pl.Float64), _to_expr(std, pl.Float64)],
            plugin_path=LIB,
            function_name="normal_expr",
            is_elementwise=True,
            kwargs={"seed": seed},
        )
    return register_plugin_function(
        args=[_length_anchor(over)],
        plugin_path=LIB,
        function_name="normal",
        is_elementwise=True,
        kwargs={"mean": mean, "std": std, "seed": seed},
    )


def _binomial_expr(
    n: IntParam,
    p: FloatParam,
    seed: int | None = None,
    over: pl.Expr | None = None,
) -> pl.Expr:
    _check_seed(seed)
    _consistent_pair(n, p, ("n", "p"))
    if isinstance(p, (int, float)):
        _check_probability(float(p))

    if _is_columnar(n):
        return register_plugin_function(
            args=[_to_expr(n, pl.UInt64), _to_expr(p, pl.Float64)],
            plugin_path=LIB,
            function_name="binomial_expr",
            is_elementwise=True,
            kwargs={"seed": seed},
        )
    return register_plugin_function(
        args=[_length_anchor(over)],
        plugin_path=LIB,
        function_name="binomial",
        is_elementwise=True,
        kwargs={"n": n, "p": p, "seed": seed},
    )


def _randint_expr(
    low: IntParam = 0,
    high: IntParam = 2,
    seed: int | None = None,
    over: pl.Expr | None = None,
) -> pl.Expr:
    _check_seed(seed)
    _consistent_pair(low, high, ("low", "high"))

    if _is_columnar(low):
        return register_plugin_function(
            args=[_to_expr(low, pl.Int64), _to_expr(high, pl.Int64)],
            plugin_path=LIB,
            function_name="randint_expr",
            is_elementwise=True,
            kwargs={"seed": seed},
        )
    return register_plugin_function(
        args=[_length_anchor(over)],
        plugin_path=LIB,
        function_name="randint",
        is_elementwise=True,
        kwargs={"low": low, "high": high, "seed": seed},
    )


def _check_size(size: int | None) -> None:
    if size is not None and size < 0:
        raise ValueError("size must be a non-negative integer")


def _eager(expr: pl.Expr, size: int) -> pl.Series:
    return pl.select(expr).to_series()


def rand(
    low: FloatParam = None,
    high: FloatParam = None,
    seed: int | None = None,
    *,
    size: int | None = None,
) -> pl.Expr | pl.Series:
    """
    Uniform `[low, high)` random draws.

    Parameters
    ----------
    low, high : float, str (column name), pl.Expr, or None
        Distribution bounds. Must both be scalars or both be column-like.
        Defaults to ``[0.0, 1.0)``.
    seed : int or None, optional
        Reproducible draws.
    size : int or None, keyword-only
        If given, eagerly evaluate and return a Series of that length.
        Otherwise returns a polars Expr to be used in a select/with_columns.

    Returns
    -------
    pl.Expr or pl.Series
    """
    _check_size(size)
    if size is None:
        return _rand_expr(low=low, high=high, seed=seed)
    over = pl.int_range(0, size).cast(pl.Float64)
    return _eager(_rand_expr(low=low, high=high, seed=seed, over=over).alias("rand"), size)


uniform = rand


def normal(
    mean: FloatParam = 0.0,
    std: FloatParam = 1.0,
    seed: int | None = None,
    *,
    size: int | None = None,
) -> pl.Expr | pl.Series:
    """
    Normal (Gaussian) random draws.

    Parameters
    ----------
    mean, std : float, str (column name), pl.Expr, or None
        Distribution parameters. Must both be scalars or both be column-like.
    seed : int or None, optional
    size : int or None, keyword-only
        If given, eagerly evaluate and return a Series of that length.

    Returns
    -------
    pl.Expr or pl.Series
    """
    _check_size(size)
    if size is None:
        return _normal_expr(mean=mean, std=std, seed=seed)
    over = pl.int_range(0, size).cast(pl.Float64)
    return _eager(_normal_expr(mean=mean, std=std, seed=seed, over=over).alias("normal"), size)


def binomial(
    n: IntParam,
    p: FloatParam,
    seed: int | None = None,
    *,
    size: int | None = None,
) -> pl.Expr | pl.Series:
    """
    Binomial random draws.

    Parameters
    ----------
    n : int, str (column name), or pl.Expr
        Number of trials.
    p : float, str (column name), or pl.Expr
        Probability of success.
    seed : int or None, optional
    size : int or None, keyword-only
        If given, eagerly evaluate and return a Series of that length.

    Returns
    -------
    pl.Expr or pl.Series
    """
    _check_size(size)
    if size is None:
        return _binomial_expr(n=n, p=p, seed=seed)
    over = pl.int_range(0, size).cast(pl.Float64)
    return _eager(_binomial_expr(n=n, p=p, seed=seed, over=over).alias("binomial"), size)


def randint(
    low: IntParam = 0,
    high: IntParam = 2,
    seed: int | None = None,
    *,
    size: int | None = None,
) -> pl.Expr | pl.Series:
    """
    Uniform random integers in ``[low, high)``.

    Parameters
    ----------
    low, high : int, str (column name), or pl.Expr
        Bounds; ``high`` is exclusive. Must both be scalars or both be column-like.
    seed : int or None, optional
    size : int or None, keyword-only
        If given, eagerly evaluate and return a Series of that length.

    Returns
    -------
    pl.Expr or pl.Series
    """
    _check_size(size)
    if size is None:
        return _randint_expr(low=low, high=high, seed=seed)
    over = pl.int_range(0, size).cast(pl.Float64)
    return _eager(_randint_expr(low=low, high=high, seed=seed, over=over).alias("randint"), size)


@pl.api.register_expr_namespace("random")
class _RandomExpr:
    """Random distributions anchored to the length of the parent expression."""

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def rand(
        self,
        low: FloatParam = None,
        high: FloatParam = None,
        seed: int | None = None,
    ) -> pl.Expr:
        return _rand_expr(low=low, high=high, seed=seed, over=self._expr)

    uniform = rand

    def normal(
        self,
        mean: FloatParam = 0.0,
        std: FloatParam = 1.0,
        seed: int | None = None,
    ) -> pl.Expr:
        return _normal_expr(mean=mean, std=std, seed=seed, over=self._expr)

    def binomial(
        self,
        n: IntParam,
        p: FloatParam,
        seed: int | None = None,
    ) -> pl.Expr:
        return _binomial_expr(n=n, p=p, seed=seed, over=self._expr)

    def randint(
        self,
        low: IntParam = 0,
        high: IntParam = 2,
        seed: int | None = None,
    ) -> pl.Expr:
        return _randint_expr(low=low, high=high, seed=seed, over=self._expr)


@pl.api.register_dataframe_namespace("random")
class Random:
    """
    Namespace for adding columns of random draws to a ``DataFrame``.

    Parameters
    ----------
    df : pl.DataFrame
        The dataframe to apply the random functions on.
    """

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def rand(
        self,
        low: FloatParam = None,
        high: FloatParam = None,
        seed: int | None = None,
        name: str | None = None,
    ) -> pl.DataFrame:
        return self._df.with_columns(
            _rand_expr(low=low, high=high, seed=seed).alias(name or "rand")
        )

    uniform = rand

    def normal(
        self,
        mean: FloatParam = 0.0,
        std: FloatParam = 1.0,
        seed: int | None = None,
        name: str | None = None,
    ) -> pl.DataFrame:
        return self._df.with_columns(
            _normal_expr(mean=mean, std=std, seed=seed).alias(name or "normal")
        )

    def binomial(
        self,
        n: IntParam,
        p: FloatParam,
        seed: int | None = None,
        name: str | None = None,
    ) -> pl.DataFrame:
        return self._df.with_columns(_binomial_expr(n=n, p=p, seed=seed).alias(name or "binomial"))

    def randint(
        self,
        low: IntParam = 0,
        high: IntParam = 2,
        seed: int | None = None,
        name: str | None = None,
    ) -> pl.DataFrame:
        return self._df.with_columns(
            _randint_expr(low=low, high=high, seed=seed).alias(name or "randint")
        )


@pl.api.register_lazyframe_namespace("random")
class _RandomLazyFrame:
    """Same API as ``df.random`` but for ``LazyFrame``."""

    def __init__(self, lf: pl.LazyFrame) -> None:
        self._lf = lf

    def rand(
        self,
        low: FloatParam = None,
        high: FloatParam = None,
        seed: int | None = None,
        name: str | None = None,
    ) -> pl.LazyFrame:
        return self._lf.with_columns(
            _rand_expr(low=low, high=high, seed=seed).alias(name or "rand")
        )

    uniform = rand

    def normal(
        self,
        mean: FloatParam = 0.0,
        std: FloatParam = 1.0,
        seed: int | None = None,
        name: str | None = None,
    ) -> pl.LazyFrame:
        return self._lf.with_columns(
            _normal_expr(mean=mean, std=std, seed=seed).alias(name or "normal")
        )

    def binomial(
        self,
        n: IntParam,
        p: FloatParam,
        seed: int | None = None,
        name: str | None = None,
    ) -> pl.LazyFrame:
        return self._lf.with_columns(_binomial_expr(n=n, p=p, seed=seed).alias(name or "binomial"))

    def randint(
        self,
        low: IntParam = 0,
        high: IntParam = 2,
        seed: int | None = None,
        name: str | None = None,
    ) -> pl.LazyFrame:
        return self._lf.with_columns(
            _randint_expr(low=low, high=high, seed=seed).alias(name or "randint")
        )
