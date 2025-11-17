This is a custom ROC-AUC implementation in C++ that uses radix sort to efficiently run on a single thread.

## Compile

To compile, run `./compile.sh`. The `CppAuc` class in `__init__.py` will call the compile script if the `cpp_auc.so` file does not exist.

## Changes
 - No support for `sample_weights` to make implementation more efficient.
 - Return type of `double` for enhanced precision.
 - Radix sort on 23-bit mantissa of floats in range [1,2).

