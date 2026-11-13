# Changelog

## 0.6.0

### Fixed

- Generate one continuous random sequence per expression. Seeded random draws
  no longer restart at every chunk when Polars partitions or streams the input.
  This fixes repeated blocks, including the roughly 8.3% unique values reported
  for a million-row draw on 12 workers.
- Seeded outputs now match across eager, lazy in-memory, and streaming evaluation
  for the same ordered inputs, including column-valued parameters and nulls.
- Mark entropy-based expressions as nondeterministic so Polars does not combine
  independent seedless draws during common subexpression elimination.

### Changed

- Require Python 3.11+ and Polars 2.1+ (below 3.0).
- Upgrade Rust Polars to 0.55.2, pyo3-polars to 0.28, and PyO3 to 0.29.
- Use Rust's default allocator to avoid recursive Python attachment during
  custom allocator initialization on macOS with the updated bindings.
- Build stable-ABI wheels for Python 3.11+ using Rust 1.95.
- Drop release wheel targets without a matching Polars 2.1 runtime wheel
  (32-bit Linux/Windows, ARMv7, and PowerPC).
- Random expressions evaluate the full input in memory, including in streaming
  queries. They no longer use the incorrect per-chunk parallel path. Streaming
  performance figures from older releases should not be used for this version.
- Check seeded outputs across benchmark scenarios before timing them.

For earlier releases, see the
[GitHub releases](https://github.com/diegoglozano/polars-random/releases).
