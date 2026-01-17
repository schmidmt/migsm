# MIGSA

[![License](https://img.shields.io/badge/license-MIT-blue)]()

**Moment Independent Global Sensitivity Analysis** — a set of tools in **Rust** and **Python** for computing global sensitivity measures using kernel density estimation (KDE) and hierarchical Dirichlet process (HDP) methods for Borgonovo δ (Delta) sensitivity indices.

## Overview

MIGSA provides efficient implementations of **moment-independent sensitivity analysis** methods.
These tools are useful for understanding the influence of input variables on model outputs *without* relying on variance-based approaches — instead using **density-based** measures to detect changes in output distributions.

The libraries include:

* **Rust core library** (`migsa`) for performant sensitivity computation
* **Python bindings** (`pymigsa`) for easy integration into data science workflows

## Features

* Compute **δ (Delta) sensitivity measures** using:

  * Kernel Density Estimation (KDE)
  * Hierarchical Dirichlet Process (HDP) density estimation
* Designed for **global sensitivity analysis**
* Works with multivariate models and complex output distributions
* High performance from Rust core and seamless Python interface

## Installation

### Rust (core library)

Add to your `Cargo.toml`:

```toml
[dependencies]
migsa = "0.x"
```

Then in your Rust code:

```rust
use migsa::sensitivity;
```

### Python (bindings)

Install from PyPI (if published):

```bash
pip install migsa
```

Or install directly from the GitHub repo:

```bash
git clone https://github.com/schmidmt/migsm.git
cd migsm
pip install ./pymigsa
```

Then import in Python:

```python
import pymigsa
```

> These examples assume tabular data with input variables and corresponding model outputs.

## Tests

To run the Rust tests:

```sh
cargo test
```

To test the Python bindings:

```sh
pytest tests/
```

## License

MIGSA is released under the **MIT License**. See [LICENSE](./LICENSE) for details.

## Citation

If you use MIGSA in academic work, consider citing it (provide citation here once available).
