from __future__ import annotations

from pathlib import Path

import polars as pl
from polars.plugins import register_plugin_function

from polars_random._internal import __version__ as VERSION

__version__ = VERSION

LIB = Path(__file__).parent

__all__ = ["sum_as_string"]


@pl.api.register_expr_namespace("random")
class Random:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def rand(self) -> pl.Expr:
        return register_plugin_function(
            args=[self._expr],
            plugin_path=LIB,
            function_name="rand",
            is_elementwise=True,
        )

    def normal(self) -> pl.Expr:
        return register_plugin_function(
            args=[self._expr],
            plugin_path=LIB,
            function_name="normal",
            is_elementwise=True,
        )
