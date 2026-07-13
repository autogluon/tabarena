Custom C++ metric kernels for fast ensemble simulation, single-threaded:

- **ROC AUC**: radix sort on the 23-bit mantissa of floats in range [1,2).
- **RMSE**: fused single-pass `sqrt(mean((a-b)^2))` (numpy needs three passes and two temporaries).

## Compile

To compile, run `./compile.sh`. The `CppMetrics` class in `__init__.py` will call the compile script if the `cpp_metrics.so` file does not exist.

## Changes vs the reference (sklearn) implementations
 - No support for `sample_weights` to make implementation more efficient.
 - Return type of `double` for enhanced precision.
